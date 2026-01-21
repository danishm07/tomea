import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from transformers.models.bert.modeling_bert import (
    BertSelfAttention,
    BertAttention,
    BertSelfOutput,
    BertIntermediate,
    BertOutput,
    BertLayer,
    BertEncoder,
    BertPooler,
    BertPreTrainedModel,
)
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, SequenceClassifierOutput
from typing import Optional, Tuple, Union
import math # Import the math module


class GatedBertSelfAttention(BertSelfAttention):
    """
    A novel mechanism from the paper: "A simple modification—applying a head-specific sigmoid gate
    after the Scaled Dot-Product Attention (SDPA)—consistently improves performance."
    This implementation applies a head-specific, multiplicative sigmoid gate after the SDPA output.
    """

    def __init__(self, config, position_embedding_type=None):
        if not hasattr(config, '_attn_implementation') or config._attn_implementation is None:
            config._attn_implementation = 'eager'
        super().__init__(config, position_embedding_type)

        # The paper specifies "head-specific sigmoid gate after the Scaled Dot-Product Attention"
        # This means the gate should operate on the output of the attention scores multiplied by values,
        # before the final concatenation and projection.
        # The output of SDPA for a single head has shape (batch_size, seq_len, head_dim).
        # For head-specific gating, each head needs its own gating parameters.
        # The gate is applied to the output of each head *before* concatenation.
        # The paper states "SDPA Elementwise G1" as the most effective, which means
        # the gate scores are vectors with the same dimensionality as Y (the SDPA output).
        # So, for each head, the gate will have parameters to produce a score of shape (seq_len, head_dim).

        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # W_theta for the gate, applied to the query output (X in the paper's Eq 5, for G1)
        # The paper states "query-dependent sparse gating scores to modulate the SDPA output."
        # This implies the gate's input (X in Eq 5) is derived from the query.
        # The SDPA output for a single head is (batch_size, seq_len, head_dim).
        # The gate needs to produce scores of the same shape.
        # If the gate is head-specific and element-wise, each head needs its own W_theta.
        # The input to the gate is the query (Q), which has shape (batch_size, seq_len, head_dim) per head.
        # So, W_theta should map from head_dim to head_dim for each head.
        self.gate_proj = nn.Linear(self.attention_head_size, self.attention_head_size)
        self.gate_activation = nn.Sigmoid()

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor]:
        # Purge unexpected arguments for compatibility
        for bad_arg in ['num_items_in_batch', 'cache_position', 'past_key_values']:
            kwargs.pop(bad_arg, None)

        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask is also from the encoder
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        if past_key_value is not None:
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            relative_position_ids = position_ids_l - position_ids_r
            position_embedding = self.distance_embedding(relative_position_ids.to(self.distance_embedding.weight.device))
            position_embedding = position_embedding.unsqueeze(0).unsqueeze(0)
            relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, position_embedding)
            attention_scores = attention_scores + relative_position_scores

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # make them receive all 0s. This is different from conventional dropout where
        # each element is randomly set to 0.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        # --- Gating Mechanism (Novel Part) ---
        # The paper states "applying a head-specific sigmoid gate after the Scaled Dot-Product Attention (SDPA)"
        # The SDPA output here is `context_layer` which has shape (batch_size, num_heads, seq_len, head_dim).
        # The gate is "elementwise" and "head-specific".
        # The input to the gate (X in Eq 5) is query-dependent.
        # We use the `query_layer` (which is head-specific and element-wise) as input to the gate.

        # Reshape query_layer for gate_proj: (batch_size * num_heads, seq_len, head_dim)
        query_for_gate = query_layer.permute(0, 2, 1, 3).reshape(-1, query_layer.size(2), self.attention_head_size)
        
        # Apply gate projection and activation
        gate_scores = self.gate_activation(self.gate_proj(query_for_gate)) # (batch_size * num_heads, seq_len, head_dim)

        # Reshape gate_scores back to (batch_size, num_heads, seq_len, head_dim)
        gate_scores = gate_scores.view(query_layer.size(0), self.num_attention_heads, query_layer.size(2), self.attention_head_size)

        # Apply multiplicative gating: Y' = Y * sigma(XW_theta)
        context_layer = context_layer * gate_scores
        # --- End Gating Mechanism ---

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


def get_model(base_model_name: str, num_labels: int):
    """
    Loads a pre-trained BERT model and injects the GatedBertSelfAttention mechanism.

    Args:
        base_model_name (str): The name of the base BERT model (e.g., "bert-base-uncased").
        num_labels (int): The number of labels for the classification head.

    Returns:
        transformers.PreTrainedModel: The modified BERT model.
    """
    config = BertConfig.from_pretrained(base_model_name)
    
    # Inject the custom attention layer into the config
    # This is a common pattern for custom layer injection in HuggingFace
    # We replace BertSelfAttention with GatedBertSelfAttention
    # The BertAttention class wraps BertSelfAttention and BertSelfOutput
    # So we need to replace the self.attention attribute within BertAttention
    
    # Load the base model
    model = BertModel.from_pretrained(base_model_name, config=config)

    # Iterate through each layer in the encoder and replace the self-attention module
    for i, layer in enumerate(model.encoder.layer):
        layer.attention.self = GatedBertSelfAttention(config)

    # For classification tasks, you might need to add a classification head
    # This example assumes a simple sequence classification head
    class BertForSequenceClassificationWithGating(BertPreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.num_labels = num_labels
            self.config = config

            self.bert = model # Use the modified BERT model
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.classifier = nn.Linear(config.hidden_size, self.num_labels)

            # Initialize weights and apply final processing
            self.post_init()

        def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            *args,
            **kwargs,
        ) -> Union[Tuple, SequenceClassifierOutput]: # Changed return type hint
            # Purge unexpected arguments for compatibility with Trainer
            for bad_arg in ['num_items_in_batch', 'cache_position', 'past_key_values']:
                kwargs.pop(bad_arg, None)

            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                *args,
                **kwargs,
            )

            pooled_output = outputs[1]

            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)

            loss = None
            if labels is not None:
                if self.config.problem_type is None:
                    if self.num_labels == 1:
                        self.config.problem_type = "regression"
                    elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                        self.config.problem_type = "single_label_classification"
                    else:
                        self.config.problem_type = "multi_label_classification"

                if self.config.problem_type == "regression":
                    loss_fct = nn.MSELoss()
                    if self.num_labels == 1:
                        loss = loss_fct(logits.squeeze(), labels.squeeze())
                    else:
                        loss = loss_fct(logits, labels)
                elif self.config.problem_type == "single_label_classification":
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                elif self.config.problem_type == "multi_label_classification":
                    loss_fct = nn.BCEWithLogitsLoss()
                    loss = loss_fct(logits, labels)

            if not return_dict:
                output = (logits,) + outputs[2:]
                return ((loss,) + output) if loss is not None else output

            # The error message "Model output format unknown: <class 'dict'>. Expected 'logits' attribute or tuple."
            # indicates that the HuggingFace Trainer expects the model's forward method to return either
            # a tuple (where the first element is typically the loss or logits) or an object that has
            # a 'logits' attribute (like SequenceClassifierOutput).
            # Your current return statement for `return_dict=True` returns a plain dictionary.
            # We need to wrap it in a `SequenceClassifierOutput` object.
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

    return BertForSequenceClassificationWithGating(config)


if __name__ == '__main__':
    # Example Usage:
    base_model_name = "bert-base-uncased"
    num_labels = 2  # Example for binary classification

    print(f"Loading base model: {base_model_name}")
    model_with_gating = get_model(base_model_name, num_labels)
    print("Model with GatedBertSelfAttention injected successfully!")

    # Verify the injection
    for i, layer in enumerate(model_with_gating.bert.encoder.layer):
        assert isinstance(layer.attention.self, GatedBertSelfAttention)
        print(f"Layer {i} attention type: {type(layer.attention.self)}")

    # Test with dummy input
    dummy_input_ids = torch.randint(0, model_with_gating.config.vocab_size, (1, 10))
    dummy_attention_mask = torch.ones((1, 10))
    dummy_labels = torch.randint(0, num_labels, (1,))

    print("\nTesting forward pass with dummy inputs...")
    # Test with labels (for training)
    output_with_labels = model_with_gating(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask, labels=dummy_labels)
    print(f"Output with labels (loss, logits): {output_with_labels.loss.shape}, {output_with_labels.logits.shape}")
    assert hasattr(output_with_labels, 'loss') and hasattr(output_with_labels, 'logits')

    # Test without labels (for inference)
    output_without_labels = model_with_gating(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask)
    print(f"Output without labels (logits): {output_without_labels.logits.shape}")
    assert not hasattr(output_without_labels, 'loss') and hasattr(output_without_labels, 'logits')

    print("\nModel architecture:")
    # print(model_with_gating) # Uncomment to see the full architecture

    # Check parameter count (optional)
    total_params = sum(p.numel() for p in model_with_gating.parameters())
    print(f"\nTotal parameters in the modified model: {total_params}")

    # Compare with original BERT model parameters (optional)
    original_model = BertModel.from_pretrained(base_model_name)
    original_total_params = sum(p.numel() for p in original_model.parameters())
    print(f"Total parameters in the original BERT model: {original_total_params}")

    # The added parameters come from `self.gate_proj` in GatedBertSelfAttention
    # For bert-base-uncased: hidden_size=768, num_attention_heads=12, head_dim=64
    # Each GatedBertSelfAttention adds: head_dim * head_dim = 64 * 64 = 4096 parameters for gate_proj
    # There are 12 layers in bert-base-uncased.
    # So, 12 * 4096 = 49152 additional parameters.
    # The classification head also adds parameters.
    # Original BertForSequenceClassification would have:
    # original_total_params + (hidden_size * num_labels + num_labels) for the classifier
    # Our model_with_gating.bert is just the encoder, so we compare its params to original_model.
    modified_bert_encoder_params = sum(p.numel() for p in model_with_gating.bert.parameters())
    print(f"Total parameters in the modified BERT encoder: {modified_bert_encoder_params}")
    print(f"Difference in encoder parameters (due to gating): {modified_bert_encoder_params - original_total_params}")
    # Expected difference: 12 * (64 * 64) = 49152
    assert modified_bert_encoder_params - original_total_params == model_with_gating.config.num_hidden_layers * (
        (model_with_gating.config.hidden_size // model_with_gating.config.num_attention_heads) ** 2
    )