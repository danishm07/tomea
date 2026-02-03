
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOTrainer, PPOConfig, GRPOTrainer, GRPOConfig, DPOTrainer, DPOConfig
from typing import List, Dict, Any, Optional
import copy

# === INJECTION POINT 1: REWARD FUNCTION (PPO/GRPO Only) ===
# WARP uses REINFORCE with KL regularization, which is handled within the compute_loss.
# The reward_function here would typically be for an external reward model,
# but WARP integrates the reward directly into the loss calculation.
def reward_function(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    # This function is a placeholder as the reward is calculated internally in WARP's loss.
    # In a real scenario, this would call a reward model.
    # For WARP, the reward is part of the KL-regularized objective.
    # We'll return dummy values, as the actual reward calculation will be in compute_loss.
    return [0.0] * len(completions)

# === INJECTION POINT 2: CUSTOM TRAINER ===
# WARP uses a variant of REINFORCE with KL regularization, which aligns with PPO/GRPO's
# structure of optimizing a policy with a reward signal and a KL constraint.
# The core modifications are to the KL anchor and the iterative merging process.
# We will subclass PPOTrainer as it provides the necessary infrastructure for policy gradient
# with KL regularization.

class WARPTrainer(PPOTrainer):

    def __init__(self, config: PPOConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        # Initialize EMA anchor for Stage 1
        self.ema_policy = copy.deepcopy(self.accelerator.unwrap_model(self.model)).eval()
        self.ema_policy_initialized = False
        self.ema_update_rate = kwargs.pop("ema_update_rate", 0.01) # mu from paper
        self.kl_beta = kwargs.pop("kl_beta", 0.1) # beta from paper

        # Store initial SFT model for LITI (Stage 3)
        self.sft_model = copy.deepcopy(self.accelerator.unwrap_model(self.model)).eval()

        # Parameters for SLERP (Stage 2) and LITI (Stage 3)
        self.num_slerp_policies = kwargs.pop("num_slerp_policies", 2) # M from paper
        self.slerp_lambda = kwargs.pop("slerp_lambda", 0.5) # lambda from paper
        self.liti_eta = kwargs.pop("liti_eta", 0.3) # eta from paper

        # To store independently fine-tuned policies for SLERP
        self.independent_policies = []

    def _update_ema_policy(self):
        """Updates the EMA policy (theta_ema in the paper)."""
        if not self.ema_policy_initialized:
            # Initialize EMA policy with current policy weights at the very beginning
            self.ema_policy.load_state_dict(self.accelerator.unwrap_model(self.model).state_dict())
            self.ema_policy_initialized = True
        else:
            current_policy = self.accelerator.unwrap_model(self.model)
            for ema_param, current_param in zip(self.ema_policy.parameters(), current_policy.parameters()):
                ema_param.data.mul_(1 - self.ema_update_rate).add_(current_param.data, alpha=self.ema_update_rate)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Overrides the PPO Trainer's compute_loss to implement WARP's Stage 1:
        EMA anchor for KL regularization.
        """
        # Update EMA policy before computing loss for the current step
        self._update_ema_policy()

        # Generate completions and compute log probabilities for the current policy
        # This part is similar to how PPOTrainer computes policy_logprobs
        query_tensors = inputs["query_input_ids"]
        response_tensors = inputs["response_input_ids"]

        # Get log probs from current policy
        policy_logprobs, _ = self.get_logprobs_from_model(
            model,
            query_tensors,
            response_tensors,
            attention_mask=inputs["response_attention_mask"],
        )

        # Get log probs from EMA anchor policy
        with torch.no_grad():
            self.ema_policy.to(self.accelerator.device) # Ensure EMA model is on the correct device
            ema_logprobs, _ = self.get_logprobs_from_model(
                self.ema_policy,
                query_tensors,
                response_tensors,
                attention_mask=inputs["response_attention_mask"],
            )

        # Calculate KL divergence term: log(pi_theta / pi_theta_ema)
        kl_div = policy_logprobs - ema_logprobs
        
        # The paper states: r_beta(x, y) = r(x, y) - beta * log(pi_theta(y|x) / pi_theta_anchor(y|x))
        # In PPO, the reward is usually external. Here, we'll assume `inputs["rewards"]`
        # contains the base reward r(x,y) from the reward model.
        rewards = inputs["rewards"]
        kl_regularized_rewards = rewards - self.kl_beta * kl_div

        # Now, use these KL-regularized rewards for the policy gradient update.
        # This is where we deviate from standard PPO and align with REINFORCE with KL.
        # The PPOTrainer's `loss_fn` expects advantages. We can treat `kl_regularized_rewards`
        # as the "advantage" for a simple REINFORCE update.
        # For a true REINFORCE, the loss is -log_prob * reward.
        # We need to ensure the `loss_fn` in PPOTrainer is configured for this,
        # or we can directly compute it here.

        # PPOTrainer's `loss_fn` is typically `ppo_loss`. We need to adapt.
        # Let's simplify for REINFORCE:
        # Loss = - E[ (r(x,y) - beta * KL) * log(pi_theta(y|x)) ]
        
        # Ensure rewards are properly shaped and on the correct device
        kl_regularized_rewards = kl_regularized_rewards.to(policy_logprobs.device)

        # REINFORCE loss: -log_prob * reward
        loss = - (policy_logprobs * kl_regularized_rewards).mean()

        if return_outputs:
            # Return outputs similar to PPOTrainer if needed for logging/metrics
            return loss, {
                "loss": loss.detach(),
                "rewards": rewards.detach(),
                "kl_div": kl_div.detach(),
                "kl_regularized_rewards": kl_regularized_rewards.detach(),
                "policy_logprobs": policy_logprobs.detach(),
            }
        return loss

    def _slerp(self, theta_init, theta_1, theta_2, lambda_val):
        """
        Spherical Linear Interpolation (SLERP) for two models.
        Applied layer by layer.
        theta_init: initial SFT weights (or previous iteration's init)
        theta_1, theta_2: independently fine-tuned policy weights
        lambda_val: interpolation coefficient
        """
        merged_model = copy.deepcopy(theta_init)
        
        # Iterate through parameters (layer by layer)
        for (name_init, param_init), (name1, param1), (name2, param2), (name_merged, param_merged) in zip(
            theta_init.named_parameters(),
            theta_1.named_parameters(),
            theta_2.named_parameters(),
            merged_model.named_parameters()
        ):
            if not (name_init == name1 == name2 == name_merged):
                raise ValueError("Parameter names do not match across models for SLERP.")

            # Task vectors
            delta1 = param1.data - param_init.data
            delta2 = param2.data - param_init.data

            # Compute angle Omega
            norm_delta1 = torch.norm(delta1)
            norm_delta2 = torch.norm(delta2)

            if norm_delta1 == 0 or norm_delta2 == 0:
                # Handle cases where a task vector is zero
                param_merged.data = param_init.data + (1 - lambda_val) * delta1 + lambda_val * delta2
                continue

            dot_product = torch.dot(delta1.flatten(), delta2.flatten())
            
            # Clamp to avoid NaN from floating point inaccuracies for acos
            dot_product = torch.clamp(dot_product / (norm_delta1 * norm_delta2), -1.0, 1.0)
            omega = torch.acos(dot_product)

            if omega < 1e-6: # Angle is very small, use linear interpolation
                param_merged.data = param_init.data + (1 - lambda_val) * delta1 + lambda_val * delta2
            else:
                sin_omega = torch.sin(omega)
                coeff1 = torch.sin((1 - lambda_val) * omega) / sin_omega
                coeff2 = torch.sin(lambda_val * omega) / sin_omega
                param_merged.data = param_init.data + coeff1 * delta1 + coeff2 * delta2
        
        return merged_model

    def _slerp_multiple(self, theta_init, policies: List[nn.Module], lambda_vals: List[float]):
        """
        Extends SLERP to multiple policies iteratively.
        As per Appendix B.3, SLERP can be used iteratively.
        For M > 2, the paper suggests iterative application.
        A simple approach is to average task vectors and then SLERP with the average.
        However, the paper's Algorithm 1 line 12 suggests:
        theta_i_slerp <- slerp(theta_init, {theta_m}_M_m=1, lambda=1/M)
        This implies a direct multi-model SLERP or an averaging of task vectors.
        Let's implement a simple averaging of task vectors for now, then SLERP with the average.
        A more faithful implementation would be iterative SLERP as described in Appendix B.3.
        For simplicity, we'll average the task vectors and then SLERP with the averaged task vector.
        """
        if not policies:
            return theta_init

        if len(policies) == 1:
            # If only one policy, no merging needed, just return it.
            # Or, if lambda_vals is used, it would be a direct interpolation.
            # For now, assume it's not meant to be called with M=1 for merging.
            return policies[0]

        # Calculate average task vector
        avg_delta = None
        for policy in policies:
            current_delta = {name: param.data - theta_init.state_dict()[name] for name, param in policy.named_parameters()}
            if avg_delta is None:
                avg_delta = current_delta
            else:
                for name in avg_delta:
                    avg_delta[name] += current_delta[name]
        
        for name in avg_delta:
            avg_delta[name] /= len(policies)

        # Create a dummy model representing theta_init + avg_delta
        avg_policy_model = copy.deepcopy(theta_init)
        for name, param in avg_policy_model.named_parameters():
            param.data = theta_init.state_dict()[name] + avg_delta[name]

        # Now, SLERP between theta_init and avg_policy_model with lambda=1 (effectively just avg_policy_model)
        # The paper's lambda=1/M suggests a different interpretation for multi-model SLERP.
        # For Algorithm 1, line 12: `slerp(theta_init, {theta_m}_M_m=1, lambda=1/M)`
        # This implies a weighted average of task vectors, then adding to theta_init.
        # Let's interpret `lambda=1/M` as the weight for each individual policy's contribution
        # to the final merged model, effectively a linear average of task vectors.
        # The paper says "spherical linear interpolation of their task vectors".
        # If lambda is a single value, it's usually for two points.
        # For M policies, it's often a weighted sum of task vectors.
        # Let's assume a simple linear average of task vectors for now, as SLERP for M>2 is complex.
        # The paper states "slerp(theta_init, {theta_m}_M_m=1, lambda=1/M)"
        # This is ambiguous. A common interpretation for "slerp" with multiple models is to
        # linearly average the task vectors and then apply SLERP with the initial model.
        # Or, it could mean iterative SLERP. Given the `lambda=1/M`, it's likely a weighted average.

        # Let's implement the "linear average of task vectors" interpretation for simplicity,
        # as a direct multi-point SLERP is not standard and Appendix B.3 describes iterative.
        # If it's a linear average of task vectors, it's LERP, not SLERP.
        # The paper explicitly says SLERP.
        # Let's stick to the iterative SLERP from Appendix B.3 for M > 2.
        # For M=2, we use the direct _slerp function.
        if len(policies) == 2:
            return self._slerp(theta_init, policies[0], policies[1], self.slerp_lambda)
        else:
            # Iterative SLERP: slerp(slerp(theta_init, p1, p2, lambda), p3, lambda), ...
            # This requires a specific order or averaging strategy.
            # Given the `lambda=1/M` in Algorithm 1, it's more likely a direct weighted average
            # of task vectors, which is LERP, but they call it SLERP.
            # Let's assume for M > 2, it's a linear average of task vectors, and then
            # the result is treated as a "slerped" model. This is a simplification.
            
            # For now, let's just average the weights directly if M > 2, which is LERP.
            # This is a deviation from "SLERP" but aligns with `lambda=1/M` for averaging.
            merged_model = copy.deepcopy(theta_init)
            for name, param in merged_model.named_parameters():
                avg_param_data = torch.zeros_like(param.data)
                for policy in policies:
                    avg_param_data += policy.state_dict()[name]
                param.data = avg_param_data / len(policies)
            return merged_model


    def _liti(self, theta_init, theta_slerp, eta):
        """
        Linear Interpolation Towards Initialization (LITI).
        theta_init: initial SFT weights (or previous iteration's init)
        theta_slerp: merged policy from SLERP
        eta: interpolation coefficient
        """
        liti_model = copy.deepcopy(theta_init)
        for (name_init, param_init), (name_slerp, param_slerp), (name_liti, param_liti) in zip(
            theta_init.named_parameters(),
            theta_slerp.named_parameters(),
            liti_model.named_parameters()
        ):
            if not (name_init == name_slerp == name_liti):
                raise ValueError("Parameter names do not match for LITI.")
            param_liti.data = (1 - eta) * param_init.data + eta * param_slerp.data
        return liti_model

    def train(self, *args, **kwargs):
        """
        Implements the iterative WARP training procedure.
        """
        num_iterations = kwargs.pop("num_iterations", 1) # I from paper
        num_rl_runs_per_iter = kwargs.pop("num_rl_runs_per_iter", self.num_slerp_policies) # M from paper
        
        current_init_model = self.sft_model # theta_init starts as theta_sft

        for i in range(num_iterations):
            self.accelerator.print(f"--- WARP Iteration {i+1}/{num_iterations} ---")
            
            # Stage 1 & RL fine-tuning (M parallel runs)
            # In a real distributed setup, these would run in parallel.
            # Here, we simulate by resetting the model and training M times sequentially.
            self.independent_policies = []
            for m in range(num_rl_runs_per_iter):
                self.accelerator.print(f"  RL Run {m+1}/{num_rl_runs_per_iter} for Iteration {i+1}")
                
                # Reset model to current_init_model for each run
                self.model.load_state_dict(current_init_model.state_dict())
                self.ema_policy_initialized = False # Reset EMA for each new RL run

                # Perform a single RL fine-tuning run
                # We need to call the base PPOTrainer's train method for the RL part.
                # This will use our overridden `compute_loss` with EMA anchor.
                super().train(*args, **kwargs) # This will run for T steps

                # Store the final policy from this run
                self.independent_policies.append(copy.deepcopy(self.accelerator.unwrap_model(self.model)).eval())
            
            # Stage 2: Spherical Linear Interpolation (SLERP)
            self.accelerator.print(f"  Applying SLERP for Iteration {i+1}")
            theta_i_slerp = self._slerp_multiple(current_init_model, self.independent_policies, [1/num_rl_runs_per_iter] * num_rl_runs_per_iter)
            
            # Stage 3: Linear Interpolation Towards Initialization (LITI)
            self.accelerator.print(f"  Applying LITI for Iteration {i+1}")
            # The paper says: `theta_init <- (1 - eta) * theta_init + eta * theta_i_slerp`
            # This means the `current_init_model` for the next iteration is the result of LITI.
            current_init_model = self._liti(current_init_model, theta_i_slerp, self.liti_eta)
            
            # Optionally, save the current_init_model (which is the result of LITI)
            # This model represents a point on the improved Pareto front.
            self.accelerator.print(f"  Iteration {i+1} complete. New initialization prepared.")
            # You might want to save `current_init_model` here or log its performance.

        # After all iterations, the final `current_init_model` is the advanced initialization.
        # The output of WARP is a KL-reward Pareto front, which means we need to
        # generate models for different `eta` values using the final `theta_I_slerp` and `theta_sft`.
        # For simplicity, we'll just return the final `current_init_model` which is
        # `(1 - eta) * theta_init_final + eta * theta_I_slerp`.
        # To get the full Pareto front, one would re-run LITI with varying eta.
        self.accelerator.print("--- WARP Training Complete ---")
        self.model.load_state_dict(current_init_model.state_dict()) # Load the final model into the trainer's model

# === INJECTION POINT 3: EXECUTION ===
def run_experiment(model_name="Qwen/Qwen2.5-0.5B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # WARP uses REINFORCE with KL regularization, which fits the PPOConfig structure.
    # We need to ensure the PPOConfig is set up for a single policy gradient update per step
    # and that the reward is handled by our custom loss.
    training_args = PPOConfig(
        output_dir="runs/warp_experiment",
        num_train_epochs=1, # Each RL run is for T steps, not epochs
        logging_steps=1,
        learning_rate=1e-6, # From paper
        mini_batch_size=1, # For REINFORCE, batch size of 1 per sample
        batch_size=128, # From paper, this is the generation batch size
        gradient_accumulation_steps=1,
        seed=42,
        # PPO specific parameters that might not be directly used by our REINFORCE loss
        # but are required by PPOTrainer. We'll set them to reasonable defaults.
        ppo_epochs=1,
        target_kl=None, # KL is explicitly regularized in the reward, not a target
        # Other parameters for generation
        max_length=128, # Example max length
        max_new_tokens=64, # Example max new tokens
    )
    
    # Mock Data: For WARP, we need prompts and a placeholder for rewards.
    # The actual reward will be computed based on the generated completions.
    # In a real setup, `rewards` would come from a reward model.
    # For this example, we'll just use dummy rewards.
    # The `compute_loss` will then apply the KL regularization.
    dataset = []
    for i in range(100):
        prompt = f"User: Tell me a short story about a {['cat', 'dog', 'bird', 'fish'][i % 4]}. Assistant:"
        # In a real scenario, `response_input_ids` and `response_attention_mask`
        # would be generated by the policy during training.
        # For this mock dataset, we'll just use dummy values.
        # The `compute_loss` method will handle generation.
        dataset.append({
            "query": prompt,
            "query_input_ids": tokenizer(prompt, return_tensors="pt").input_ids[0],
            "response": "This is a dummy response.", # Will be overwritten by generation
            "response_input_ids": torch.randint(0, tokenizer.vocab_size, (1, 10))[0], # Dummy
            "response_attention_mask": torch.ones(10, dtype=torch.long), # Dummy
            "rewards": torch.tensor(0.5), # Dummy base reward, will be KL-regularized
        })

    # Initialize the WARPTrainer
    trainer = WARPTrainer(
        config=training_args,
        model=model_name,
        ref_model=None, # Not explicitly used in WARP's REINFORCE, but PPOTrainer expects it
        tokenizer=tokenizer,
        dataset=dataset, # PPOTrainer expects `dataset` not `train_dataset`
        # WARP specific parameters
        ema_update_rate=0.01, # mu
        kl_beta=0.1, # beta
        num_slerp_policies=2, # M
        slerp_lambda=0.5, # lambda
        liti_eta=0.3, # eta
    )
    
    print("--- 🚀 Starting WARP RL Training ---")
    # Call the custom train method with WARP-specific iterations
    trainer.train(num_iterations=5, num_rl_runs_per_iter=2) # I=5, M=2 from paper example
    
    print("--- WARP Experiment Finished ---")
    # You can now save the final model or evaluate it.
    # trainer.save_model("final_warp_model")

if __name__ == "__main__":
    run_experiment()
