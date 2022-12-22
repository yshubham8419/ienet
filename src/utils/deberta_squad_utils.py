import json
import numpy as np
import re
import math
import unicodedata
import six
import collections
import logging
import time
import string
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
    "PrelimPrediction",
    ["feature_index", "start_index", "end_index",
     "start_log_prob", "end_log_prob", "null_odds"])

_NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
    "NbestPrediction", ["text", "start_log_prob", "end_log_prob", "null_odds"])


class SquadExample(object):
    """A single training/test example for simple sequence classification.
     For examples without an answer, the start and end position are -1.
  """

    def __init__(self,
                 qas_id,
                 question_text,
                 paragraph_text,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=False):
        self.qas_id = qas_id
        self.question_text = question_text
        self.paragraph_text = paragraph_text
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index=None,
                 tok_start_to_orig_index=None,
                 tok_end_to_orig_index=None,
                 token_is_max_context=None,
                 tokens=None,
                 input_ids=None,
                 input_mask=None,
                 segment_ids=None,
                 paragraph_len=None,
                 p_mask=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None,
                 syntax_mask=None,
                 mask_start=None,
                 mask_end=None,
                 ):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tok_start_to_orig_index = tok_start_to_orig_index
        self.tok_end_to_orig_index = tok_end_to_orig_index
        self.token_is_max_context = token_is_max_context
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.paragraph_len = paragraph_len
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.p_mask = p_mask
        self.syntax_mask = syntax_mask
        self.mask_start = mask_start
        self.mask_end = mask_end


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def read_squad_examples(input_file, is_training):
    """Read a SQuAD json file into a list of SquadExample."""


    examples = []
    pos_count = 0
    neg_count = 0
    for entry in input_file:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                orig_answer_text = None
                is_impossible = False
                if is_training:
                    is_impossible = qa.get("is_impossible", False)
                    if (len(qa["answers"]) != 1) and (not is_impossible):
                        raise ValueError(
                            "For training, each question should have exactly 1 answer.")
                    if not is_impossible:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        start_position = answer["answer_start"]
                        pos_count += 1
                    else:
                        neg_count += 1
                        start_position = -1
                        orig_answer_text = ""

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    paragraph_text=paragraph_text,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    is_impossible=is_impossible)

                examples.append(example)

    logger.info("pos%d, neg%d"%(pos_count, neg_count))

    return examples

def preprocess_text(inputs, remove_space=True, lower=False):
  """preprocess data by removing extra space and normalize data."""
  outputs = inputs
  if remove_space:
    outputs = " ".join(inputs.strip().split())

  if six.PY2 and isinstance(outputs, str):
    try:
      outputs = six.ensure_text(outputs, "utf-8")
    except UnicodeDecodeError:
      outputs = six.ensure_text(outputs, "latin-1")

  outputs = unicodedata.normalize("NFKD", outputs)
  outputs = "".join([c for c in outputs if not unicodedata.combining(c)])
  if lower:
    outputs = outputs.lower()

  return outputs


def _convert_index(index, pos, m=None, is_start=True):
    """Converts index."""
    if index[pos] is not None:
        return index[pos]
    n = len(index)
    rear = pos
    while rear < n - 1 and index[rear] is None:
        rear += 1
    front = pos
    while front > 0 and index[front] is None:
        front -= 1
    assert index[front] is not None or index[rear] is not None
    if index[front] is None:
        if index[rear] >= 1:
            if is_start:
                return 0
            else:
                return index[rear] - 1
        return index[rear]
    if index[rear] is None:
        if m is not None and index[front] < m - 1:
            if is_start:
                return index[front] + 1
            else:
                return m - 1
        return index[front]
    if is_start:
        if index[rear] > index[front] + 1:
            return index[front] + 1
        else:
            return index[rear]
    else:
        if index[rear] > index[front] + 1:
            return index[rear] - 1
        else:
            return index[front]


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training,
                                 do_lower_case, use_plausible_answer=False, debug=False):
    """Loads a data file into a list of `InputBatch`s."""

    cnt_pos, cnt_neg = 0, 0
    unique_id = 1000000000
    max_n, max_m = 1024, 1024
    f = np.zeros((max_n, max_m), dtype=np.float32)

    features = []
    for (example_index, example) in enumerate(examples):

        if example_index % 100 == 0:
            logger.info("Converting {}/{} pos {} neg {}".format(
                example_index, len(examples), cnt_pos, cnt_neg))
        if debug:
            if example_index == 15:
                break

        query_pieces = tokenizer.tokenize(
            preprocess_text(
                example.question_text, lower=do_lower_case))

        para_text = preprocess_text(example.paragraph_text, lower=do_lower_case)

        para_tokens = tokenizer.tokenize(para_text)

        para_tokens = [six.ensure_text(token, "utf-8") for token in para_tokens]
        query_pieces = [six.ensure_text(token, "utf-8") for token in query_pieces]

        char_cnt = 0
        tok_start_to_chartok_index = []
        tok_end_to_chartok_index = []
        chartok_to_tok_index = []
        prcocessed_text = []

        for i, token in enumerate(para_tokens):
            if token.startswith("##"):
                new_token = token[2:]
            else:
                new_token = " {0}".format(token)

            prcocessed_text.append(new_token)
            chartok_to_tok_index.extend([i] * len(new_token))
            tok_start_to_chartok_index.append(char_cnt)
            char_cnt += len(new_token)
            tok_end_to_chartok_index.append(char_cnt - 1)

        tok_cat_text = "".join(prcocessed_text)
        n, m = len(example.paragraph_text), len(tok_cat_text)

        if n > max_n or m > max_m:
            max_n = max(n, max_n)
            max_m = max(m, max_m)
            f = np.zeros((max_n, max_m), dtype=np.float32)

        g = {}

        def _lcs_match(max_dist, n=n, m=m):
            """Longest-common-substring algorithm."""
            f.fill(0)
            g.clear()

            ### longest common sub sequence
            # f[i, j] = max(f[i - 1, j], f[i, j - 1], f[i - 1, j - 1] + match(i, j))
            for i in range(n):

                # note(zhiliny):
                # unlike standard LCS, this is specifically optimized for the setting
                # because the mismatch between sentence pieces and original text will
                # be small
                for j in range(i - max_dist, i + max_dist):
                    if j >= m or j < 0: continue

                    if i > 0:
                        g[(i, j)] = 0
                        f[i, j] = f[i - 1, j]

                    if j > 0 and f[i, j - 1] > f[i, j]:
                        g[(i, j)] = 1
                        f[i, j] = f[i, j - 1]

                    f_prev = f[i - 1, j - 1] if i > 0 and j > 0 else 0
                    if (preprocess_text(
                            example.paragraph_text[i], lower=do_lower_case,
                            remove_space=False) == tok_cat_text[j]
                            and f_prev + 1 > f[i, j]):
                        g[(i, j)] = 2
                        f[i, j] = f_prev + 1

        max_dist = abs(n - m) + 5
        start = time.time()
        for _ in range(2):
            _lcs_match(max_dist)
            if f[n - 1, m - 1] > 0.8 * n: break
            max_dist *= 2
        end = time.time()
        # logger.info(end-start)

        orig_to_chartok_index = [None] * n
        chartok_to_orig_index = [None] * m
        i, j = n - 1, m - 1
        while i >= 0 and j >= 0:
            if (i, j) not in g: break
            if g[(i, j)] == 2:
                orig_to_chartok_index[i] = j
                chartok_to_orig_index[j] = i
                i, j = i - 1, j - 1
            elif g[(i, j)] == 1:
                j = j - 1
            else:
                i = i - 1

        if (all(v is None for v in orig_to_chartok_index) or
                f[n - 1, m - 1] < 0.8 * n):
            print("MISMATCH DETECTED!")
            continue

        tok_start_to_orig_index = []
        tok_end_to_orig_index = []
        for i in range(len(para_tokens)):
            start_chartok_pos = tok_start_to_chartok_index[i]
            end_chartok_pos = tok_end_to_chartok_index[i]
            start_orig_pos = _convert_index(chartok_to_orig_index, start_chartok_pos,
                                            n, is_start=True)
            end_orig_pos = _convert_index(chartok_to_orig_index, end_chartok_pos,
                                          n, is_start=False)

            tok_start_to_orig_index.append(start_orig_pos)
            tok_end_to_orig_index.append(end_orig_pos)

        if not is_training:
            tok_start_position = tok_end_position = None
        if not use_plausible_answer:
            if is_training and example.is_impossible:
                tok_start_position = 0
                tok_end_position = 0

            if is_training and not example.is_impossible:
                start_position = example.start_position
                end_position = start_position + len(example.orig_answer_text) - 1

                start_chartok_pos = _convert_index(orig_to_chartok_index, start_position,
                                                   is_start=True)
                tok_start_position = chartok_to_tok_index[start_chartok_pos]

                end_chartok_pos = _convert_index(orig_to_chartok_index, end_position,
                                                 is_start=False)
                tok_end_position = chartok_to_tok_index[end_chartok_pos]
                if tok_start_position > tok_end_position or example.orig_answer_text != example.paragraph_text[
                                                                                        start_position: end_position + 1]:
                    continue

        else:
            if is_training:
                start_position = example.start_position
                end_position = start_position + len(example.orig_answer_text) - 1

                start_chartok_pos = _convert_index(orig_to_chartok_index, start_position,
                                                   is_start=True)
                tok_start_position = chartok_to_tok_index[start_chartok_pos]

                end_chartok_pos = _convert_index(orig_to_chartok_index, end_position,
                                                 is_start=False)
                tok_end_position = chartok_to_tok_index[end_chartok_pos]
                if tok_start_position > tok_end_position or example.orig_answer_text != example.paragraph_text[
                                                                                        start_position: end_position + 1]:
                    continue

        def _piece_to_id(x):
            return tokenizer.convert_tokens_to_ids(x)

        all_doc_tokens = _piece_to_id(para_tokens)

        if len(query_pieces) > max_query_length:
            query_tokens = _piece_to_id(query_pieces[0:max_query_length])
        else:
            query_tokens = _piece_to_id(query_pieces)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        syntax_pas = None
        syntax_que = None

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_is_max_context = {}
            segment_ids = []
            p_mask = []
            syntax_mask = []
            mask_start = []
            mask_end = []

            cur_tok_start_to_orig_index = []
            cur_tok_end_to_orig_index = []

            tokens.append(_piece_to_id(["[CLS]"])[0])
            segment_ids.append(0)
            p_mask.append(1)
            mask_start.append(1)
            mask_end.append(1)

            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
                p_mask.append(0)
                mask_start.append(0)
                mask_end.append(0)
            tokens.append(_piece_to_id(["[SEP]"])[0])
            segment_ids.append(0)
            p_mask.append(0)
            mask_start.append(0)
            mask_end.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i

                cur_tok_start_to_orig_index.append(
                    tok_start_to_orig_index[split_token_index])
                cur_tok_end_to_orig_index.append(
                    tok_end_to_orig_index[split_token_index])

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
                p_mask.append(1)

                if not para_tokens[split_token_index].startswith("##"):
                    mask_start.append(1)
                else:
                    mask_start.append(0)

                if i == doc_span.length - 1:
                    mask_end.append(1)
                elif not para_tokens[split_token_index + 1].startswith("##"):
                    mask_end.append(1)
                else:
                    mask_end.append(0)

            tokens.append(_piece_to_id(["[SEP]"])[0])
            segment_ids.append(1)
            p_mask.append(0)
            mask_start.append(0)
            mask_end.append(0)
            paragraph_len = len(tokens)
            input_ids = tokens


            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)
            while len(input_ids) < max_seq_length:
                input_ids.append(_piece_to_id("[PAD]"))
                input_mask.append(0)
                segment_ids.append(_piece_to_id("[PAD]"))
                p_mask.append(_piece_to_id("[PAD]"))
                mask_start.append(_piece_to_id("[PAD]"))
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            span_is_impossible = example.is_impossible
            start_position = None
            end_position = None
            if is_training:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and
                        tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    # continue
                    start_position = 0
                    end_position = 0
                    span_is_impossible = True
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

                if span_is_impossible and not use_plausible_answer:
                    start_position = 0
                    end_position = 0

            if is_training:
                feat_example_index = None
            else:
                feat_example_index = example_index

            feature = InputFeatures(
                unique_id=unique_id,
                example_index=feat_example_index,
                doc_span_index=doc_span_index,
                tok_start_to_orig_index=cur_tok_start_to_orig_index,
                tok_end_to_orig_index=cur_tok_end_to_orig_index,
                token_is_max_context=token_is_max_context,
                tokens=tokenizer.convert_ids_to_tokens(tokens),
                input_ids=input_ids,
                input_mask=input_mask,
                mask_start=mask_start,
                mask_end=mask_end,
                segment_ids=segment_ids,
                paragraph_len=paragraph_len,
                start_position=start_position,
                end_position=end_position,
                is_impossible=span_is_impossible,
                p_mask=p_mask,
                syntax_mask=syntax_mask
            )

            # Run callback

            features.append(feature)
            unique_id += 1
            if span_is_impossible:
                cnt_neg += 1
            else:
                cnt_pos += 1


    logger.info("Total number of instances: {} = pos {} neg {}".format(
        cnt_pos + cnt_neg, cnt_pos, cnt_neg))
    return features


def accumulate_predictions_v2(result_dict, all_examples,
                              all_features, all_results,
                              max_answer_length, nbest_size):

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)
    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result
    for (example_index, example) in enumerate(all_examples):
        score_null = 1000000
        if example_index not in result_dict:
            result_dict[example_index] = {}
        features = example_index_to_features[example_index]
        for (feature_index, feature) in enumerate(features):
            if feature.unique_id not in result_dict[example_index]:
                result_dict[example_index][feature.unique_id] = {}
            result = unique_id_to_result[feature.unique_id]
            doc_offset = feature.tokens.index("[CLS]") + 1
            start_indexes = _get_best_indexes(result.start_logits, nbest_size)
            end_indexes = _get_best_indexes(result.end_logits, nbest_size)
            feature_null_score = result.start_logits[0] + result.end_logits[0]
            if feature_null_score < score_null:
                score_null = feature_null_score
                min_null_feature_index = feature_index
                null_start_logit = result.start_logits[0]
                null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index - doc_offset >= len(feature.tok_start_to_orig_index):
                        continue
                    if start_index - doc_offset < 0:
                        continue
                    if end_index - doc_offset >= len(feature.tok_end_to_orig_index):
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1

                    if length > max_answer_length:
                        continue
                    start_idx = start_index - doc_offset
                    end_idx = end_index - doc_offset
                    if (start_idx, end_idx) not in result_dict[example_index][feature.unique_id]:
                        result_dict[example_index][feature.unique_id][(start_idx, end_idx)] = []
                    result_dict[example_index][feature.unique_id][(start_idx, end_idx)].append(
                        (result.start_logits[start_index], result.end_logits[end_index], feature_null_score))

def write_predictions_v2(result_dict, all_examples, all_features,
                         all_results, n_best_size, max_answer_length,
                         output_prediction_file,
                         output_nbest_file, output_null_log_odds_file,
                         null_score_diff_threshold):
    logger.info("Writing predictions to: %s" % (output_prediction_file))
    logger.info("Writing nbest to: %s" % (output_nbest_file))
    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()
    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]
        prelim_predictions = []
        for (feature_index, feature) in enumerate(features):
            for ((start_idx, end_idx), logprobs) in \
                    result_dict[example_index][feature.unique_id].items():
                start_log_prob = 0
                end_log_prob = 0
                null_odds = 0
                for logprob in logprobs:
                    start_log_prob += logprob[0]
                    end_log_prob += logprob[1]
                    null_odds += logprob[2]
                prelim_predictions.append(
                    _PrelimPrediction(
                        feature_index=feature_index,
                        start_index=start_idx,
                        end_index=end_idx,
                        start_log_prob=start_log_prob / len(logprobs),
                        end_log_prob=end_log_prob / len(logprobs),
                        null_odds=null_odds / len(logprobs)))

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_log_prob + x.end_log_prob - x.null_odds),
            reverse=True)
        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index >= 0:

                tok_start_to_orig_index = feature.tok_start_to_orig_index
                tok_end_to_orig_index = feature.tok_end_to_orig_index
                start_orig_pos = tok_start_to_orig_index[pred.start_index]
                end_orig_pos = tok_end_to_orig_index[pred.end_index]

                paragraph_text = example.paragraph_text
                final_text = paragraph_text[start_orig_pos: end_orig_pos + 1].strip()

                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_log_prob=pred.start_log_prob,
                    end_log_prob=pred.end_log_prob,
                    null_odds=pred.null_odds
                ))

            # In very rare edge cases we could have no valid predictions. So we
            # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(
                    text="",
                    start_log_prob=-1e6,
                    end_log_prob=-1e6,
                    null_odds=-100
                ))

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_log_prob + entry.end_log_prob)
            if not best_non_null_entry:
                best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_log_prob"] = entry.start_log_prob
            output["end_log_prob"] = entry.end_log_prob
            output["null_odds"] = entry.null_odds
            nbest_json.append(output)

        assert len(nbest_json) >= 1
        assert best_non_null_entry is not None

        score_diff = best_non_null_entry.null_odds
        # score_diff = sum(cls_dict[example_index]) / len(cls_dict[example_index])
        scores_diff_json[example.qas_id] = score_diff
        # predict null answers when null threshold is provided
        if null_score_diff_threshold is None or score_diff < null_score_diff_threshold:
            all_predictions[example.qas_id] = best_non_null_entry.text
        else:
            all_predictions[example.qas_id] = ""

        all_nbest_json[example.qas_id] = nbest_json
        assert len(nbest_json) >= 1

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    with open(output_null_log_odds_file, "w") as writer:
        writer.write(json.dumps(scores_diff_json, indent=4) + "\n")
    return all_predictions, scores_diff_json


def make_qid_to_has_ans(dataset):
    qid_to_has_ans = {}
    for article in dataset:
        for p in article['paragraphs']:
            for qa in p['qas']:
                qid_to_has_ans[qa['id']] = bool(qa['answers'])
    return qid_to_has_ans


def normalize_answer_v2(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact(a_gold, a_pred):
    return int(normalize_answer_v2(a_gold) == normalize_answer_v2(a_pred))


def get_tokens(s):
    if not s: return []
    return normalize_answer_v2(s).split()


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_raw_scores(dataset, preds):
    exact_scores = {}
    f1_scores = {}
    for article in dataset:
        for p in article['paragraphs']:
            for qa in p['qas']:
                qid = qa['id']
                gold_answers = [a['text'] for a in qa['answers']
                                if normalize_answer_v2(a['text'])]
                if not gold_answers:
                    # For unanswerable questions, only correct answer is empty string
                    gold_answers = ['']
                if qid not in preds:
                    print('Missing prediction for %s' % qid)
                    continue
                a_pred = preds[qid]
                # Take max over all gold answers
                exact_scores[qid] = max(compute_exact(a, a_pred) for a in gold_answers)
                f1_scores[qid] = max(compute_f1(a, a_pred) for a in gold_answers)
    return exact_scores, f1_scores


def apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh):
    new_scores = {}
    for qid, s in scores.items():
        pred_na = na_probs[qid] > na_prob_thresh
        if pred_na:
            new_scores[qid] = float(not qid_to_has_ans[qid])
        else:
            new_scores[qid] = s
    return new_scores


def make_eval_dict(exact_scores, f1_scores, qid_list=None):
    if not qid_list:
        total = len(exact_scores)
        return collections.OrderedDict([
            ('exact', 100.0 * sum(exact_scores.values()) / total),
            ('f1', 100.0 * sum(f1_scores.values()) / total),
            ('total', total),
        ])
    else:
        total = len(qid_list)
        return collections.OrderedDict([
            ('exact', 100.0 * sum(exact_scores[k] for k in qid_list) / total),
            ('f1', 100.0 * sum(f1_scores[k] for k in qid_list) / total),
            ('total', total),
        ])


def find_best_thresh(preds, scores, na_probs, qid_to_has_ans):
    num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
    cur_score = num_no_ans
    best_score = cur_score
    best_thresh = 0.0
    qid_list = sorted(na_probs, key=lambda k: na_probs[k])
    for i, qid in enumerate(qid_list):
        if qid not in scores: continue
        if qid_to_has_ans[qid]:
            diff = scores[qid]
        else:
            if preds[qid]:
                diff = -1
            else:
                diff = 0
        cur_score += diff
        if cur_score > best_score:
            best_score = cur_score
            best_thresh = na_probs[qid]
    return 100.0 * best_score / len(scores), best_thresh


def find_all_best_thresh(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans):
    best_exact, exact_thresh = find_best_thresh(preds, exact_raw, na_probs, qid_to_has_ans)
    best_f1, f1_thresh = find_best_thresh(preds, f1_raw, na_probs, qid_to_has_ans)
    main_eval['best_exact'] = best_exact
    main_eval['best_exact_thresh'] = exact_thresh
    main_eval['best_f1'] = best_f1
    main_eval['best_f1_thresh'] = f1_thresh


def evaluate_v2(result_dict, prediction_json, eval_examples,
                eval_features, all_results, n_best_size, max_answer_length,
                output_prediction_file, output_nbest_file,
                output_null_log_odds_file):
    null_score_diff_threshold = None
    predictions, na_probs = write_predictions_v2(
        result_dict, eval_examples, eval_features,
        all_results, n_best_size, max_answer_length,
        output_prediction_file, output_nbest_file,
        output_null_log_odds_file, null_score_diff_threshold)

    na_prob_thresh = 1.0  # default value taken from the eval script
    qid_to_has_ans = make_qid_to_has_ans(prediction_json)  # maps qid to True/False
    exact_raw, f1_raw = get_raw_scores(prediction_json, predictions)
    exact_thresh = apply_no_ans_threshold(exact_raw, na_probs, qid_to_has_ans,
                                          na_prob_thresh)
    f1_thresh = apply_no_ans_threshold(f1_raw, na_probs, qid_to_has_ans,
                                       na_prob_thresh)
    out_eval = make_eval_dict(exact_thresh, f1_thresh)
    find_all_best_thresh(out_eval, predictions, exact_raw, f1_raw, na_probs, qid_to_has_ans)
    null_score_diff_threshold = out_eval["best_f1_thresh"]

    predictions, na_probs = write_predictions_v2(
        result_dict, eval_examples, eval_features,
        all_results, n_best_size, max_answer_length,
        output_prediction_file, output_nbest_file,
        output_null_log_odds_file, null_score_diff_threshold)

    qid_to_has_ans = make_qid_to_has_ans(prediction_json)  # maps qid to True/False
    exact_raw, f1_raw = get_raw_scores(prediction_json, predictions)
    exact_thresh = apply_no_ans_threshold(exact_raw, na_probs, qid_to_has_ans,
                                          na_prob_thresh)
    f1_thresh = apply_no_ans_threshold(f1_raw, na_probs, qid_to_has_ans,
                                       na_prob_thresh)
    out_eval = make_eval_dict(exact_thresh, f1_thresh)
    out_eval["null_score_diff_threshold"] = null_score_diff_threshold
    return out_eval


def accumulate_predictions_v2j(result_dict, all_examples,
                              all_features, all_results,
                              max_answer_length, start_n_top, end_n_top, reverse=False):
    """accumulate predictions for each positions in a dictionary."""

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    for (example_index, example) in enumerate(all_examples):
        if example_index not in result_dict:
            result_dict[example_index] = {}
        features = example_index_to_features[example_index]

        for (feature_index, feature) in enumerate(features):
            if feature.unique_id not in result_dict[example_index]:
                result_dict[example_index][feature.unique_id] = {}
            result = unique_id_to_result[feature.unique_id]
            if not reverse:
                doc_offset = feature.tokens.index("[SEP]") + 1
            else:
                doc_offset = feature.tokens.index("[CLS]") + 1
            for i in range(start_n_top):
                for j in range(end_n_top):
                    start_log_prob = result.start_top_log_probs[i]
                    start_index = result.start_top_index[i]

                    cur_null_score = result.cls_logits

                    j_index = i * end_n_top + j

                    end_log_prob = result.end_top_log_probs[j_index]
                    end_index = result.end_top_index[j_index]
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index - doc_offset >= len(feature.tok_start_to_orig_index):
                        continue
                    if start_index - doc_offset < 0:
                        continue
                    if end_index - doc_offset >= len(feature.tok_end_to_orig_index):
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    start_idx = start_index - doc_offset
                    end_idx = end_index - doc_offset
                    if (start_idx, end_idx) not in result_dict[example_index][feature.unique_id]:
                        result_dict[example_index][feature.unique_id][(start_idx, end_idx)] = []
                    result_dict[example_index][feature.unique_id][(start_idx, end_idx)].append(
                        (start_log_prob, end_log_prob, cur_null_score))


def write_predictions_v2j(result_dict, all_examples, all_features,
                         all_results, n_best_size, max_answer_length,
                         null_score_diff_threshold):
    """Write final predictions to the json file and log-odds of null if needed."""


    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []

        for (feature_index, feature) in enumerate(features):
            for ((start_idx, end_idx), logprobs) in \
                    result_dict[example_index][feature.unique_id].items():
                start_log_prob = 0
                end_log_prob = 0
                null_odds = 0
                start_null_odds = 0
                end_null_odds = 0
                for logprob in logprobs:
                    start_log_prob += logprob[0]
                    end_log_prob += logprob[1]
                    null_odds += logprob[2]

                prelim_predictions.append(
                    _PrelimPrediction(
                        feature_index=feature_index,
                        start_index=start_idx,
                        end_index=end_idx,
                        start_log_prob=start_log_prob / len(logprobs),
                        end_log_prob=end_log_prob / len(logprobs),
                        null_odds=null_odds / len(logprobs),
                    ))

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_log_prob + x.end_log_prob - x.null_odds),
            reverse=True)

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index >= 0:

                tok_start_to_orig_index = feature.tok_start_to_orig_index
                tok_end_to_orig_index = feature.tok_end_to_orig_index
                start_orig_pos = tok_start_to_orig_index[pred.start_index]
                end_orig_pos = tok_end_to_orig_index[pred.end_index]

                paragraph_text = example.paragraph_text
                final_text = paragraph_text[start_orig_pos: end_orig_pos + 1].strip()

                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_log_prob=pred.start_log_prob,
                    end_log_prob=pred.end_log_prob,
                    null_odds=pred.null_odds,
                ))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(
                    text="",
                    start_log_prob=-1e6,
                    end_log_prob=-1e6,
                    null_odds=-100
                ))

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_log_prob + entry.end_log_prob)
            if not best_non_null_entry:
                best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_log_prob"] = entry.start_log_prob
            output["end_log_prob"] = entry.end_log_prob
            output["null_odds"] = entry.null_odds
            nbest_json.append(output)

        assert len(nbest_json) >= 1
        assert best_non_null_entry is not None

        score_diff = best_non_null_entry.null_odds
        # score_diff = sum(cls_dict[example_index]) / len(cls_dict[example_index])
        scores_diff_json[example.qas_id] = score_diff
        # predict null answers when null threshold is provided
        if null_score_diff_threshold is None or score_diff < null_score_diff_threshold:
            all_predictions[example.qas_id] = best_non_null_entry.text
        else:
            all_predictions[example.qas_id] = ""

        all_nbest_json[example.qas_id] = nbest_json
        assert len(nbest_json) >= 1

    return all_nbest_json
