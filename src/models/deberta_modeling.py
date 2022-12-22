import torch
import logging
from transformers import DebertaV2PreTrainedModel, DebertaV2Model, DebertaModel, DebertaPreTrainedModel
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DebertaForQAJointly(DebertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, cls_drop_prob=0.1, topk=5, entropy_beta=0.0):
        super().__init__(config)
        logger.info("using cls drop prob of %f" % cls_drop_prob)
        if entropy_beta > 0.0:
            logger.info("using entropy_beta of %f" % entropy_beta)
        self.topk = topk
        self.deberta = DebertaModel(config)
        self.start_layer = torch.nn.Linear(config.hidden_size, 1)
        self.end_layer = torch.nn.Sequential(
            torch.nn.Linear(2*config.hidden_size, config.hidden_size),
            torch.nn.Linear(config.hidden_size, 1))
        self.classifier_layer = torch.nn.Sequential(
            torch.nn.Linear(2*config.hidden_size, config.hidden_size),
            torch.nn.GELU(),
            torch.nn.Dropout(p=cls_drop_prob),
            torch.nn.Linear(config.hidden_size, 1, bias=False))
        self.entropy_beta=entropy_beta


    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                inputs_embeds=None,
                start_positions=None,
                end_positions=None,
                is_impossible=None,
                p_mask=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                ):
        training = start_positions is not None
        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        shape = sequence_output.size()
        bsz = shape[0]
        seq_len = shape[1]
        hidden_size = shape[2]
        start_logits = self.start_layer(sequence_output)
        start_logits = torch.squeeze(start_logits, dim=-1)
        start_logits_masked = start_logits * p_mask - 1e30 * (1 - p_mask)
        start_log_probs = torch.nn.functional.log_softmax(start_logits_masked, dim=-1)
        start_probs = torch.nn.functional.softmax(start_logits_masked, dim=-1)
        if training:
            # getting end_logits jointly with start information, training
            start_index = torch.nn.functional.one_hot(start_positions, num_classes=seq_len)
            start_features = torch.unsqueeze(start_index, -1) * sequence_output
            start_features = torch.sum(start_features, dim=1)
            start_features = torch.tile(torch.unsqueeze(start_features, dim=1), [1, seq_len, 1])
            end_logits = torch.cat((start_features, sequence_output), -1)
            end_logits = self.end_layer(end_logits)
            end_logits = torch.squeeze(end_logits, -1)
            end_logits_masked = end_logits * p_mask - 1e30 * (1 - p_mask)
            end_log_probs = torch.nn.functional.log_softmax(end_logits_masked, dim=-1)
            end_probs = torch.nn.functional.softmax(end_logits_masked, dim=-1)
            # getting cls_logits jointly with start information
        else:
            start_top_log_probs, start_top_index = torch.topk(start_log_probs, k=self.topk)
            end_logits = torch.tile(torch.unsqueeze(sequence_output, 1),
                                    [1, self.topk, 1, 1])
            start_index = torch.nn.functional.one_hot(start_top_index, seq_len)
            start_features = torch.sum(
                torch.unsqueeze(sequence_output, 1) * torch.unsqueeze(start_index, -1), dim=-2
            )

            start_features = torch.tile(torch.unsqueeze(start_features, dim=2),
                                        [1, 1, seq_len, 1])
            end_logits = torch.cat((start_features, end_logits), -1)
            end_logits = self.end_layer(end_logits)
            end_logits = torch.squeeze(end_logits, -1)
            ex_p_mask = p_mask[:, None]
            end_logits_masked = end_logits * ex_p_mask - 1e30 * (1 - ex_p_mask)
            end_log_probs = torch.nn.functional.log_softmax(end_logits_masked, dim=-1)
            end_top_log_probs, end_top_index = torch.topk(end_log_probs, k=self.topk)
            end_top_log_probs = torch.reshape(end_top_log_probs, [-1, self.topk * self.topk])
            end_top_index = torch.reshape(end_top_index, [-1, self.topk * self.topk])

        # getting cls_logits jointly with start information
        cls_feature = sequence_output[:, 0, :]
        start_p = torch.nn.functional.softmax(start_logits_masked, dim=-1)
        start_fea = torch.einsum("blh,bl->bh", sequence_output, start_p)
        ans_feature = torch.cat((start_fea, cls_feature), dim=-1)
        cls_logits = self.classifier_layer(ans_feature)
        cls_logits = torch.squeeze(cls_logits, dim=-1)

        # computing loss
        total_loss = None
        if training:
            span_loss_fn = CrossEntropyLoss()
            start_loss = span_loss_fn(start_logits_masked, start_positions)
            end_loss = span_loss_fn(end_logits_masked, end_positions)

            if self.entropy_beta > 0:
                # using entropy penalty
                start_entropy = compute_neg_entropy(start_log_probs, start_probs)
                end_entropy = compute_neg_entropy(end_log_probs, end_probs)
                start_loss += start_entropy * self.entropy_beta
                end_loss += end_entropy * self.entropy_beta

            span_loss = (start_loss + end_loss) * 0.5
            ans_loss_fn = BCEWithLogitsLoss()
            ans_loss = ans_loss_fn(cls_logits, is_impossible)
            total_loss = span_loss + ans_loss

        output = (start_log_probs, end_log_probs, cls_logits)

        if training:
            output = (total_loss, start_log_probs, end_log_probs, cls_logits)
        else:
            output = (start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits)

        return output


class DebertaV2ForQAJointly(DebertaV2PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, cls_drop_prob=0.1, topk=5, entropy_beta=0.0):
        super().__init__(config)
        logger.info("using cls drop prob of %f" % cls_drop_prob)
        if entropy_beta > 0.0:
            logger.info("using entropy_beta of %f" % entropy_beta)
        self.topk = topk
        self.deberta = DebertaV2Model(config)
        self.start_layer = torch.nn.Linear(config.hidden_size, 1)
        self.end_layer = torch.nn.Sequential(
            torch.nn.Linear(2 * config.hidden_size, config.hidden_size),
            torch.nn.Linear(config.hidden_size, 1))
        self.classifier_layer = torch.nn.Sequential(
            torch.nn.Linear(2 * config.hidden_size, config.hidden_size),
            torch.nn.GELU(),
            torch.nn.Dropout(p=cls_drop_prob),
            torch.nn.Linear(config.hidden_size, 1, bias=False))
        self.entropy_beta = entropy_beta

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                inputs_embeds=None,
                start_positions=None,
                end_positions=None,
                is_impossible=None,
                p_mask=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                ):
        training = start_positions is not None
        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        shape = sequence_output.size()
        bsz = shape[0]
        seq_len = shape[1]
        hidden_size = shape[2]
        start_logits = self.start_layer(sequence_output)
        start_logits = torch.squeeze(start_logits, dim=-1)
        start_logits_masked = start_logits * p_mask - 1e30 * (1 - p_mask)
        start_log_probs = torch.nn.functional.log_softmax(start_logits_masked, dim=-1)
        start_probs = torch.nn.functional.softmax(start_logits_masked, dim=-1)
        if training:
            # getting end_logits jointly with start information, training
            start_index = torch.nn.functional.one_hot(start_positions, num_classes=seq_len)
            start_features = torch.unsqueeze(start_index, -1) * sequence_output
            start_features = torch.sum(start_features, dim=1)
            start_features = torch.tile(torch.unsqueeze(start_features, dim=1), [1, seq_len, 1])
            end_logits = torch.cat((start_features, sequence_output), -1)
            end_logits = self.end_layer(end_logits)
            end_logits = torch.squeeze(end_logits, -1)
            end_logits_masked = end_logits * p_mask - 1e30 * (1 - p_mask)
            end_log_probs = torch.nn.functional.log_softmax(end_logits_masked, dim=-1)
            end_probs = torch.nn.functional.softmax(end_logits_masked, dim=-1)
            # getting cls_logits jointly with start information
        else:
            start_top_log_probs, start_top_index = torch.topk(start_log_probs, k=self.topk)
            end_logits = torch.tile(torch.unsqueeze(sequence_output, 1),
                                    [1, self.topk, 1, 1])
            start_index = torch.nn.functional.one_hot(start_top_index, seq_len)
            start_features = torch.sum(
                torch.unsqueeze(sequence_output, 1) * torch.unsqueeze(start_index, -1), dim=-2
            )

            start_features = torch.tile(torch.unsqueeze(start_features, dim=2),
                                        [1, 1, seq_len, 1])
            end_logits = torch.cat((start_features, end_logits), -1)
            end_logits = self.end_layer(end_logits)
            end_logits = torch.squeeze(end_logits, -1)
            ex_p_mask = p_mask[:, None]
            end_logits_masked = end_logits * ex_p_mask - 1e30 * (1 - ex_p_mask)
            end_log_probs = torch.nn.functional.log_softmax(end_logits_masked, dim=-1)
            end_top_log_probs, end_top_index = torch.topk(end_log_probs, k=self.topk)
            end_top_log_probs = torch.reshape(end_top_log_probs, [-1, self.topk * self.topk])
            end_top_index = torch.reshape(end_top_index, [-1, self.topk * self.topk])

        # getting cls_logits jointly with start information
        cls_feature = sequence_output[:, 0, :]
        start_p = torch.nn.functional.softmax(start_logits_masked, dim=-1)
        start_fea = torch.einsum("blh,bl->bh", sequence_output, start_p)
        ans_feature = torch.cat((start_fea, cls_feature), dim=-1)
        cls_logits = self.classifier_layer(ans_feature)
        cls_logits = torch.squeeze(cls_logits, dim=-1)

        # computing loss
        total_loss = None
        if training:
            span_loss_fn = CrossEntropyLoss()
            start_loss = span_loss_fn(start_logits_masked, start_positions)
            end_loss = span_loss_fn(end_logits_masked, end_positions)

            if self.entropy_beta > 0:
                # using entropy penalty
                start_entropy = compute_neg_entropy(start_log_probs, start_probs)
                end_entropy = compute_neg_entropy(end_log_probs, end_probs)
                start_loss += start_entropy * self.entropy_beta
                end_loss += end_entropy * self.entropy_beta

            span_loss = (start_loss + end_loss) * 0.5
            ans_loss_fn = BCEWithLogitsLoss()
            ans_loss = ans_loss_fn(cls_logits, is_impossible)
            total_loss = span_loss + ans_loss

        output = (start_log_probs, end_log_probs, cls_logits)

        if training:
            output = (total_loss, start_log_probs, end_log_probs, cls_logits)
        else:
            output = (start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits)

        return output


def compute_neg_entropy(log_probs, probs):
    neg_entropy = torch.sum(log_probs * probs, dim=-1)
    neg_entropy = torch.mean(neg_entropy, dim=-1)
    return neg_entropy
