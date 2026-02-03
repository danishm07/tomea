import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoConfig, PreTrainedModel
import math

# Ensure trl and peft are installed if you plan to use them with this model
# from trl import DPOTrainer, PPOTrainer, GRPOTrainer
# from peft import LoraConfig, get_peft_model

# Helper function for discretization (Zero-Order Hold)
def discretize_zoh(delta, A, B):
    """
    Discretizes continuous parameters (delta, A, B) to discrete (A_bar, B_bar) using ZOH.
    A: (D, N) or (B, L, D, N)
    B: (D, N) or (B, L, D, N)
    delta: (D) or (B, L, D)
    """
    # Ensure delta has an extra dimension for broadcasting with A and B
    if delta.dim() == 1: # (D)
        delta_expanded = delta.unsqueeze(-1) # (D, 1)
    elif delta.dim() == 3: # (B, L, D)
        delta_expanded = delta.unsqueeze(-1) # (B, L, D, 1)
    else:
        raise ValueError(f"Unsupported delta shape: {delta.shape}")

    # A_bar = exp(delta * A)
    A_bar = torch.exp(delta_expanded * A)

    # B_bar = (delta * A)^-1 * (exp(delta * A) - I) * delta * B
    # For diagonal A, A_inv = 1/A
    # (delta * A)^-1 * (exp(delta * A) - I) * delta
    # = (1/A) * (exp(delta * A) - I)
    # This is only if A is diagonal and non-zero.
    # The paper uses (delta * A)^-1 * (exp(delta * A) - I) * delta * B
    # For diagonal A, (delta * A)^-1 is 1/(delta*A)
    # Let's assume A is diagonal and handle the inverse carefully.
    # If A is a matrix, this requires matrix inverse. The paper states A is structured, most popular form is diagonal.
    # Assuming A is diagonal and represented by N numbers.
    # If A is (D, N) or (B, L, D, N), it means A_diag is N numbers for each D channel.
    
    # For diagonal A, A is effectively a vector of N elements for each D channel.
    # So A_diag is (D, N) or (B, L, D, N)
    
    # (delta * A)
    delta_A = delta_expanded * A # (D, N) or (B, L, D, N)

    # (exp(delta * A) - I)
    exp_delta_A_minus_I = A_bar - 1.0 # (D, N) or (B, L, D, N)

    # (delta * A)^-1 * (exp(delta * A) - I)
    # Handle potential division by zero if delta_A has zeros
    # A common numerical stable way is to use a Taylor expansion for small values or a specific function like expm1_over_x
    # For now, let's assume delta_A is non-zero.
    # If delta_A is zero, (exp(delta_A) - I) / delta_A approaches 1.
    
    # A more robust way for (exp(x) - 1) / x is to use torch.expm1(x) / x
    # Let x = delta_A
    # B_bar_coeff = torch.expm1(delta_A) / delta_A
    
    # To avoid division by zero for delta_A, we can use a small epsilon or a more sophisticated function.
    # For simplicity, let's use the direct division and assume delta_A is not zero.
    # A common trick for (exp(x)-1)/x is to use a custom function or handle small values.
    # For now, direct division:
    B_bar_coeff = exp_delta_A_minus_I / delta_A
    
    # B_bar = B_bar_coeff * delta * B
    B_bar = B_bar_coeff * delta_expanded * B # (D, N) or (B, L, D, N)

    return A_bar, B_bar


class SelectiveSSM(nn.Module):
    def __init__(self, d_model, n_state, expand_factor, **kwargs):
        super().__init__()
        self.d_model = d_model # D in the paper
        self.n_state = n_state # N in the paper
        self.expand_factor = expand_factor # E in the paper

        # Parameters A, B, C, Delta
        # A is a parameter, (D, N)
        self.A = nn.Parameter(torch.randn(d_model, n_state))

        # sB(x), sC(x), sDelta(x) are linear projections
        # Input x is (B, L, D)
        # sB(x) -> (B, L, N)
        self.sB = nn.Linear(d_model, n_state)
        # sC(x) -> (B, L, N)
        self.sC = nn.Linear(d_model, n_state)
        # sDelta(x) -> (B, L, 1) (then broadcasted to D)
        self.sDelta = nn.Linear(d_model, 1)

        # Delta parameter (D)
        self.delta_param = nn.Parameter(torch.randn(d_model))

        # tau_Delta = softplus
        self.tau_Delta = nn.Softplus()

    def forward(self, x, **kwargs):
        # x: (B, L, D)
        batch_size, seq_len, d_model = x.shape

        # 1. A: (D, N) is a parameter, already defined
        A = self.A

        # 2. B: (B, L, N) <- sB(x)
        B_selective = self.sB(x)

        # 3. C: (B, L, N) <- sC(x)
        C_selective = self.sC(x)

        # 4. Delta: (B, L, D) <- tau_Delta(delta_param + sDelta(x))
        # sDelta(x) is (B, L, 1)
        sDelta_x = self.sDelta(x) # (B, L, 1)
        # delta_param is (D)
        # Need to broadcast delta_param to (B, L, D) to add to sDelta_x
        delta_param_broadcast = self.delta_param.view(1, 1, d_model).expand(batch_size, seq_len, d_model)
        
        # The paper says sDelta(x) = BroadcastD(Linear1(x))
        # This means Linear1(x) is (B, L, 1) and then it's broadcasted to (B, L, D)
        # So, we should expand sDelta_x to (B, L, D) before adding delta_param
        sDelta_x_broadcast = sDelta_x.expand(batch_size, seq_len, d_model) # (B, L, D)

        delta = self.tau_Delta(delta_param_broadcast + sDelta_x_broadcast) # (B, L, D)

        # 5. Discretize: A_bar, B_bar: (B, L, D, N) <- discretize(delta, A, B_selective)
        # A is (D, N), delta is (B, L, D), B_selective is (B, L, N)
        # To make A compatible with delta and B_selective for element-wise operations in discretize_zoh:
        # A needs to be (B, L, D, N)
        A_expanded = A.view(1, 1, d_model, self.n_state).expand(batch_size, seq_len, d_model, self.n_state)
        # B_selective needs to be (B, L, D, N)
        # The paper says B is (B, L, N) and C is (B, L, N).
        # In discretize_zoh, B is used with A and delta.
        # The equation for B_bar is (delta * A)^-1 * (exp(delta * A) - I) * delta * B
        # This implies B should be (D, N) or (B, L, D, N) to match A.
        # Let's assume B_selective is broadcasted across D channels.
        B_selective_expanded = B_selective.unsqueeze(2).expand(batch_size, seq_len, d_model, self.n_state) # (B, L, D, N)

        A_bar, B_bar = discretize_zoh(delta, A_expanded, B_selective_expanded) # A_bar, B_bar: (B, L, D, N)

        # 6. SSM Recurrence (Scan)
        # y: (B, L, D)
        # h_t = A_bar_t * h_{t-1} + B_bar_t * x_t
        # y_t = C_selective_t * h_t
        
        # Initialize hidden state h_0
        h = torch.zeros(batch_size, d_model, self.n_state, device=x.device) # (B, D, N)
        y_outputs = []

        # C_selective is (B, L, N)
        # Need to expand C_selective to (B, L, D, N) to multiply with h (B, D, N)
        # C_selective_t for a given t is (B, N)
        # h_t for a given t is (B, D, N)
        # y_t = C_selective_t @ h_t.T (or similar)
        # The paper says C is (1, N) for LTI, and (B, L, N) for selective.
        # y_t = C_t * h_t implies C_t is (B, D, N) and h_t is (B, D, N) and element-wise product then sum over N.
        # Or C_t is (B, N) and h_t is (B, D, N) and C_t is broadcasted across D.
        # Given C is (B, L, N), it's likely broadcasted across D.
        
        for t in range(seq_len):
            # A_bar_t: (B, D, N)
            A_bar_t = A_bar[:, t, :, :]
            # B_bar_t: (B, D, N)
            B_bar_t = B_bar[:, t, :, :]
            # x_t: (B, D)
            x_t = x[:, t, :]
            # C_selective_t: (B, N)
            C_selective_t = C_selective[:, t, :]

            # h_t = A_bar_t * h_{t-1} + B_bar_t * x_t
            # x_t needs to be (B, D, 1) for multiplication with B_bar_t (B, D, N)
            x_t_expanded = x_t.unsqueeze(-1) # (B, D, 1)
            
            h = A_bar_t * h + B_bar_t * x_t_expanded # (B, D, N)

            # y_t = C_selective_t * h_t
            # C_selective_t is (B, N)
            # h is (B, D, N)
            # This implies C_selective_t is broadcasted across D, then element-wise product, then sum over N.
            y_t = torch.sum(C_selective_t.unsqueeze(1) * h, dim=-1) # (B, D)
            y_outputs.append(y_t)

        y = torch.stack(y_outputs, dim=1) # (B, L, D)

        return y


class MambaBlock(nn.Module):
    def __init__(self, d_model, n_state, expand_factor, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.expand_factor = expand_factor
        self.d_inner = d_model * expand_factor

        # Linear projection for input x
        self.in_proj = nn.Linear(d_model, self.d_inner * 2) # x and z branches

        # Convolution layer (local convolution, not global)
        # The paper mentions "standard local convolution" in H3, and Mamba simplifies this.
        # It's not explicitly detailed in the Mamba block diagram, but typically a 1D conv is used.
        # Let's assume a simple 1D convolution for now, as it's common in SSM architectures.
        # The paper's diagram shows "Conv" before SSM.
        # Let's use a small kernel size, e.g., 3.
        self.conv1d = nn.Conv1d(in_channels=self.d_inner, out_channels=self.d_inner, kernel_size=3, padding=1, groups=self.d_inner)

        # Selective SSM layer
        self.ssm = SelectiveSSM(d_model=self.d_inner, n_state=n_state, expand_factor=expand_factor)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model)

        # Activation function (SiLU / Swish)
        self.act = nn.SiLU()

    def forward(self, x, **kwargs):
        # x: (B, L, D)
        residual = x

        # Input projection
        # x_and_z: (B, L, D_inner * 2)
        x_and_z = self.in_proj(x)
        x_branch, z_branch = x_and_z.chunk(2, dim=-1) # x_branch, z_branch: (B, L, D_inner)

        # Local convolution
        # Permute for Conv1d: (B, D_inner, L)
        x_conv = self.conv1d(x_branch.transpose(1, 2)).transpose(1, 2) # (B, L, D_inner)

        # Activation
        x_act = self.act(x_conv)

        # Selective SSM
        x_ssm = self.ssm(x_act) # (B, L, D_inner)

        # Gating with z_branch
        gated_output = x_ssm * self.act(z_branch) # (B, L, D_inner)

        # Output projection
        output = self.out_proj(gated_output) # (B, L, D)

        # Residual connection
        output = output + residual

        return output


class MambaModel(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        self.d_model = config.d_model
        self.n_state = config.n_state # N in the paper, state dimension of SSM
        self.expand_factor = config.expand_factor # E in the paper
        self.num_layers = config.num_layers
        self.num_labels = config.num_labels # For sequence classification

        # Embedding layer (e.g., for tokens)
        # Assuming input is token IDs, need an embedding layer
        # If input is already features, this can be skipped or adapted.
        # For language modeling, usually vocab_size and hidden_size
        self.embedding = nn.Embedding(config.vocab_size, self.d_model)

        # Mamba blocks
        self.layers = nn.ModuleList([
            MambaBlock(self.d_model, self.n_state, self.expand_factor)
            for _ in range(self.num_layers)
        ])

        # Normalization layer (optional, as per paper)
        self.norm = nn.LayerNorm(self.d_model)

        # Classifier head for sequence classification
        self.classifier = nn.Linear(self.d_model, self.num_labels)

        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # input_ids: (B, L)
        # attention_mask: (B, L) - not directly used by Mamba's recurrence, but useful for padding
        
        # Embeddings
        x = self.embedding(input_ids) # (B, L, D)

        # Mamba blocks
        for layer in self.layers:
            x = layer(x)

        # Normalization
        x = self.norm(x) # (B, L, D)

        # For sequence classification, typically take the last token's representation
        # or average pooling. Let's take the last token for simplicity, similar to BERT.
        # Or, if it's a causal model, the representation of the last actual token.
        # For classification, usually we pool or take the [CLS] token.
        # Let's assume we take the representation corresponding to the last non-padded token.
        
        # If attention_mask is provided, find the last non-padded token
        if attention_mask is not None:
            # Get the index of the last non-padded token for each sequence
            sequence_lengths = attention_mask.sum(dim=1) - 1 # (B,)
            # Gather the last token's hidden state
            # x_cls = x[torch.arange(x.size(0)), sequence_lengths] # (B, D)
            
            # A more robust way for variable sequence lengths:
            # Create a mask for the last token of each sequence
            last_token_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
            for i, length in enumerate(sequence_lengths):
                if length >= 0: # Ensure sequence is not empty
                    last_token_mask[i, length] = True
            
            x_pooled = x[last_token_mask] # (B, D)
        else:
            # If no attention mask, assume all sequences are full length and take the last token
            x_pooled = x[:, -1, :] # (B, D)

        logits = self.classifier(x_pooled) # (B, num_labels)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits}


def get_model(base_model_name: str, num_labels: int):
    # Load a dummy config or create a custom one
    # For a real scenario, you might load a pre-existing config or define all parameters.
    config = AutoConfig.from_pretrained(base_model_name)
    
    # Override/add Mamba-specific parameters
    config.d_model = getattr(config, "hidden_size", 768) # Default to 768 if not specified
    config.n_state = 16 # N in the paper, a common choice for SSM state dimension
    config.expand_factor = 2 # E in the paper
    config.num_layers = getattr(config, "num_hidden_layers", 12) # Default to 12 layers
    config.num_labels = num_labels
    config.vocab_size = getattr(config, "vocab_size", 30522) # Default to BERT vocab size

    # Instantiate the MambaModel
    model = MambaModel(config)
    return model

