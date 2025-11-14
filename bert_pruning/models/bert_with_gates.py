"""
BERT with Hard Concrete Gates for Attention Head Pruning
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, BertConfig
from transformers.models.bert.modeling_bert import BertSelfAttention
from typing import Optional, Tuple
from .concrete_gate import ConcreteGate


class BertSelfAttentionWithGate(BertSelfAttention):
    """BERT Self-Attention with Hard Concrete Gate."""

    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type=position_embedding_type)

        # Add concrete gate for attention heads
        self.gate = ConcreteGate(
            num_heads=config.num_attention_heads,
            temperature=0.33,
            stretch_limits=(-0.1, 1.1),
            l0_penalty=1.0,
            hard=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.attention_head_size, dtype=torch.float32))
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        # Apply Hard Concrete Gate to attention heads
        # context_layer shape: [batch, num_heads, seq_len, head_dim]
        context_layer = self.gate(context_layer, is_train=self.training)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class BertModelWithGates(BertModel):
    """BERT Model with Hard Concrete Gates for head pruning."""

    def __init__(self, config: BertConfig, add_pooling_layer: bool = True):
        super().__init__(config, add_pooling_layer=add_pooling_layer)

        # Replace all self-attention layers with gated versions
        for layer in self.encoder.layer:
            old_self_attn = layer.attention.self
            layer.attention.self = BertSelfAttentionWithGate(
                config, position_embedding_type=getattr(config, "position_embedding_type", "absolute")
            )
            # Copy weights from old attention
            layer.attention.self.query.load_state_dict(old_self_attn.query.state_dict())
            layer.attention.self.key.load_state_dict(old_self_attn.key.state_dict())
            layer.attention.self.value.load_state_dict(old_self_attn.value.state_dict())

    def get_all_gates(self):
        """Return list of all ConcreteGate modules."""
        gates = []
        for layer in self.encoder.layer:
            gates.append(layer.attention.self.gate)
        return gates

    def get_total_l0_penalty(self) -> torch.Tensor:
        """Compute total L0 penalty across all layers."""
        total_penalty = 0.0
        for gate in self.get_all_gates():
            total_penalty += gate.get_penalty()
        return total_penalty

    def get_sparsity_info(self):
        """Get detailed sparsity information for all layers."""
        info = []
        for layer_idx, gate in enumerate(self.get_all_gates()):
            gate_values = gate.get_gate_values()
            sparsity_rate = gate.get_sparsity_rate().item()
            info.append({
                'layer': layer_idx,
                'gate_values': gate_values.cpu().numpy().tolist(),
                'sparsity_rate': sparsity_rate,
                'num_active_heads': int(sparsity_rate * gate.num_heads)
            })
        return info

    def prune_heads(self, threshold: float = 0.01):
        """
        Prune attention heads based on gate values.

        Args:
            threshold: Heads with gate values below this are pruned

        Returns:
            Dictionary with pruning information
        """
        pruning_info = {
            'total_heads': 0,
            'pruned_heads': 0,
            'heads_to_prune': {}
        }

        for layer_idx, gate in enumerate(self.get_all_gates()):
            head_mask = gate.prune_heads(threshold)
            heads_to_prune = [i for i, keep in enumerate(head_mask) if keep == 0]

            pruning_info['total_heads'] += gate.num_heads
            pruning_info['pruned_heads'] += len(heads_to_prune)

            if heads_to_prune:
                pruning_info['heads_to_prune'][layer_idx] = heads_to_prune

        pruning_info['pruning_ratio'] = pruning_info['pruned_heads'] / pruning_info['total_heads']

        return pruning_info


class BertForSequenceClassificationWithGates(BertPreTrainedModel):
    """BERT for sequence classification with prunable attention heads."""

    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModelWithGates(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

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
        l0_coefficient: float = 0.0,
    ):
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

            # Add L0 penalty to loss
            if l0_coefficient > 0 and self.training:
                l0_penalty = self.bert.get_total_l0_penalty()
                loss = loss + l0_coefficient * l0_penalty

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        from transformers.modeling_outputs import SequenceClassifierOutput
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_sparsity_info(self):
        """Get sparsity information from BERT model."""
        return self.bert.get_sparsity_info()

    def prune_heads(self, threshold: float = 0.01):
        """Prune heads based on gate values."""
        return self.bert.prune_heads(threshold)
