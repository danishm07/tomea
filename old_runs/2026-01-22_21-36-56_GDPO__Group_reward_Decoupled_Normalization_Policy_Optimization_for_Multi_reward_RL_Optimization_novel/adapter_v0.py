import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from transformers.models.bert.modeling_bert import BertSelfAttention, BertAttention, BertIntermediate, BertOutput, BertLayer
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, SequenceClassifierOutput
import math # Import math module at the top level

class GDPOBertSelfAttention(BertSelfAttention):
    """
    A novel self-attention mechanism inspired by GDPO's decoupled normalization concept.
    Instead of normalizing the combined attention scores, this implementation
    hypothetically applies a form of 'decoupled normalization' to the query, key, and value
    projections before computing attention scores. This is a conceptual interpretation
    as GDPO primarily deals with reward normalization in RL, not directly with attention.

    The core idea is to apply a layer normalization-like operation to Q, K, V
    independently *before* the dot product, aiming to preserve individual signal
    characteristics, analogous to GDPO preserving individual reward signals.
    """
    def __init__(self, config, position_embedding_type=None):
        if not hasattr(config, '_attn_implementation') or config._attn_implementation is None:
            config._attn_implementation = 'eager'
        super().__init__(config, position_embedding_type)
        self.query_norm = nn.LayerNorm(self.all_head_size, eps=config.layer_norm_eps)
        self.key_norm = nn.LayerNorm(self.all_head_size, eps=config.layer_norm_eps)
        self.value_norm = nn.LayerNorm(self.all_head_size, eps=config.layer_norm_eps)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        past_key_value: tuple[tuple[torch.FloatTensor]] | None = None,
        output_attentions: bool | None = False,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor]:
        for bad_arg in ['num_items_in_batch', 'cache_position', 'past_key_values']:
            kwargs.pop(bad_arg, None)

        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask is also from an encoder
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        # Apply decoupled normalization before transposing for scores
        query_layer = self.transpose_for_scores(self.query_norm(mixed_query_layer))
        key_layer = self.transpose_for_scores(self.key_norm(mixed_key_layer))
        value_layer = self.transpose_for_scores(self.value_norm(mixed_value_layer))

        if past_key_value is not None:
            if len(past_key_value) != 0:
                key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
                value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # make them receive all zero attention.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

class GDPOBertForSequenceClassification(BertModel):
    """
    A BERT model with GDPOBertSelfAttention for sequence classification.
    This class wraps the BertModel and adds a classification head,
    ensuring the output format is compatible with typical classification tasks.
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        # Replace BertSelfAttention with GDPOBertSelfAttention
        for i, layer in enumerate(self.encoder.layer):
            layer.attention.self = GDPOBertSelfAttention(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> SequenceClassifierOutput | BaseModelOutputWithPoolingAndCrossAttentions:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = super().forward(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True, # Always return dict internally for easier processing
        )

        pooled_output = outputs.pooler_output
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
                loss = loss_fct(logits, labels.float())

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def get_model(base_model_name: str, num_labels: int):
    """
    Loads a pre-trained BERT model and replaces its BertSelfAttention layers
    with GDPOBertSelfAttention, then adds a classification head.
    """
    config = BertConfig.from_pretrained(base_model_name)
    config.num_labels = num_labels

    # Instantiate the custom model which handles the attention replacement and classification head
    model = GDPOBertForSequenceClassification.from_pretrained(base_model_name, config=config)

    return model

if __name__ == '__main__':
    # Example usage:
    base_model_name = "bert-base-uncased"
    num_labels = 10

    print(f"Loading model '{base_model_name}' and injecting GDPOBertSelfAttention...")
    model = get_model(base_model_name, num_labels)
    print("Model loaded and modified successfully!")

    # Verify the replacement
    assert isinstance(model.encoder.layer[0].attention.self, GDPOBertSelfAttention)
    print("Verification successful: BertSelfAttention replaced with GDPOBertSelfAttention.")

    # Test a forward pass
    input_ids = torch.randint(0, model.config.vocab_size, (2, 128)) # Batch size 2, sequence length 128
    attention_mask = torch.ones(2, 128)
    token_type_ids = torch.zeros(2, 128)
    labels = torch.randint(0, num_labels, (2,)) # Example labels for classification

    print("\nTesting forward pass...")
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)

    print("Forward pass successful!")
    print("Output logits shape:", outputs.logits.shape)
    print("Loss:", outputs.loss.item())

    # Test with output_attentions=True
    print("\nTesting forward pass with output_attentions=True...")
    with torch.no_grad():
        outputs_attn = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_attentions=True, labels=labels)
    print("Forward pass with attentions successful!")
    print("Output logits shape:", outputs_attn.logits.shape)
    print("Attention weights shape (first layer):", outputs_attn.attentions[0].shape)
    print("Loss:", outputs_attn.loss.item())

    # Test without labels (inference mode)
    print("\nTesting forward pass without labels (inference mode)...")
    with torch.no_grad():
        outputs_inference = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    print("Inference forward pass successful!")
    print("Output logits shape:", outputs_inference.logits.shape)
    assert outputs_inference.loss is None
