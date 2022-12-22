import tensorflow as tf
import os
import six
import pickle
import collections
import numpy as np
import json
import math
import models.out4_modeling as modeling
from tensorflow.contrib import layers as contrib_layers
from utils.data import read_squad_examples
from utils.data import FeatureWriter
from utils.data import InputFeatures
from utils import tokenization
from utils import squad_utils

BATCH_SIZE = 8
MAX_ANSWER_LENGTH = 30
START_N = 5
END_N = 5
N_BEST = 20

_NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
    "NbestPrediction", ["text", "start_log_prob", "end_log_prob", "null_odds"])

_PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
    "PrelimPrediction",
    ["feature_index", "start_index", "end_index",
     "start_log_prob", "end_log_prob", "null_odds"])

RawResultV2 = collections.namedtuple(
    "RawResultV2",
    ["unique_id", "start_top_log_probs", "start_top_index",
     "end_top_log_probs", "end_top_index", "cls_logits"])


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
                                 output_fn, do_lower_case, generate_sg=False):
    """Loads a data file into a list of `InputBatch`s."""

    cnt_pos, cnt_neg = 0, 0
    unique_id = 1000000000
    max_n, max_m = 1024, 1024
    f = np.zeros((max_n, max_m), dtype=np.float32)

    for (example_index, example) in enumerate(examples):

        if example_index % 100 == 0:
            tf.logging.info("Converting {}/{} pos {} neg {}".format(
                example_index, len(examples), cnt_pos, cnt_neg))

        query_pieces = tokenization.encode_pieces(
            tokenizer.sp_model,
            tokenization.preprocess_text(
                example.question_text, lower=do_lower_case),
            return_unicode=False)

        paragraph_text = example.paragraph_text
        para_tokens = tokenization.encode_pieces(
            tokenizer.sp_model,
            tokenization.preprocess_text(
                example.paragraph_text, lower=do_lower_case),
            return_unicode=False)

        para_tokens = [six.ensure_text(token, "utf-8") for token in para_tokens]
        query_pieces = [six.ensure_text(token, "utf-8") for token in query_pieces]

        chartok_to_tok_index = []
        tok_start_to_chartok_index = []
        tok_end_to_chartok_index = []
        char_cnt = 0

        for i, token in enumerate(para_tokens):
            new_token = six.ensure_text(token).replace(
                tokenization.SPIECE_UNDERLINE.decode("utf-8"), " ")
            chartok_to_tok_index.extend([i] * len(new_token))
            tok_start_to_chartok_index.append(char_cnt)
            char_cnt += len(new_token)
            tok_end_to_chartok_index.append(char_cnt - 1)

        tok_cat_text = "".join(para_tokens).replace(
            tokenization.SPIECE_UNDERLINE.decode("utf-8"), " ")
        n, m = len(paragraph_text), len(tok_cat_text)

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
                    if (tokenization.preprocess_text(
                            paragraph_text[i], lower=do_lower_case,
                            remove_space=False) == tok_cat_text[j]
                            and f_prev + 1 > f[i, j]):
                        g[(i, j)] = 2
                        f[i, j] = f_prev + 1

        max_dist = abs(n - m) + 5
        for _ in range(2):
            _lcs_match(max_dist)
            if f[n - 1, m - 1] > 0.8 * n: break
            max_dist *= 2

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
            tf.logging.info("MISMATCH DETECTED!")
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
            if not tok_start_position <= tok_end_position:
                tf.logging.info("End before Start DETECTED!")
                continue

        def _piece_to_id(x):
            if six.PY2 and isinstance(x, six.text_type):
                x = six.ensure_binary(x, "utf-8")
            return tokenizer.sp_model.PieceToId(x)

        all_doc_tokens = list(map(_piece_to_id, para_tokens))

        if len(query_pieces) > max_query_length:
            query_tokens = list(map(_piece_to_id, query_pieces[0:max_query_length]))
        else:
            query_tokens = list(map(_piece_to_id, query_pieces))

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

            tokens.append(tokenizer.sp_model.PieceToId("[CLS]"))
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
            tokens.append(tokenizer.sp_model.PieceToId("[SEP]"))
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

                if para_tokens[split_token_index].startswith(tokenization.SPIECE_UNDERLINE.decode("utf-8")):
                    mask_start.append(1)
                else:
                    mask_start.append(0)

                if i == doc_span.length - 1:
                    mask_end.append(1)
                elif para_tokens[split_token_index + 1].startswith(tokenization.SPIECE_UNDERLINE.decode("utf-8")):
                    mask_end.append(1)
                else:
                    mask_end.append(0)

            tokens.append(tokenizer.sp_model.PieceToId("[SEP]"))
            segment_ids.append(1)
            p_mask.append(0)
            mask_start.append(0)
            mask_end.append(0)
            paragraph_len = len(tokens)
            input_ids = tokens

            input_mask = [1] * len(input_ids)

            if generate_sg:
                syntax_mask.extend([1] * len(input_ids))
                if syntax_que is not None:
                    for item in syntax_que:
                        cur_vector = [0] * len(input_ids)
                        for sparse_idx in item:
                            cur_vector[sparse_idx + 1] = 1
                        syntax_mask.extend(cur_vector)
                else:
                    for _ in range(len(query_tokens)):
                        syntax_mask.extend([1] * len(input_ids))

                syntax_mask.extend([0] * len(input_ids))
                query_offset = len(query_tokens) + 2

                if syntax_pas is not None:
                    syntax_pas_span = syntax_pas[doc_span.start:doc_span.start + doc_span.length]
                    for item in syntax_pas_span:
                        cur_vector = [0] * len(input_ids)
                        for sparse_idx in item:
                            loc = sparse_idx + query_offset - doc_span.start
                            if len(cur_vector) > loc >= query_offset:
                                cur_vector[loc] = 1
                        syntax_mask.extend(cur_vector)
                else:
                    for _ in range(len(doc_span.length)):
                        syntax_mask.extend([1] * len(input_ids))
                syntax_mask.extend([0] * len(input_ids))
                assert len(syntax_mask) == len(input_ids) ** 2

            ##

            span_is_impossible = example.is_impossible
            start_position = None
            end_position = None
            if is_training and not span_is_impossible:
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

            if is_training and span_is_impossible:
                start_position = 0
                end_position = 0

            if example_index < 20:
                tf.logging.info("*** Example ***")
                tf.logging.info("unique_id: %s" % (unique_id))
                tf.logging.info("example_index: %s" % (example_index))
                tf.logging.info("doc_span_index: %s" % (doc_span_index))
                tf.logging.info("tok_start_to_orig_index: %s" % " ".join(
                    [str(x) for x in cur_tok_start_to_orig_index]))
                tf.logging.info("tok_end_to_orig_index: %s" % " ".join(
                    [str(x) for x in cur_tok_end_to_orig_index]))
                tf.logging.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in six.iteritems(token_is_max_context)
                ]))
                tf.logging.info("input_pieces: %s" % " ".join(
                    [tokenizer.sp_model.IdToPiece(x) for x in tokens]))
                tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                tf.logging.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                tf.logging.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

                if is_training and span_is_impossible:
                    tf.logging.info("impossible example span")

                if is_training and not span_is_impossible:
                    pieces = [tokenizer.sp_model.IdToPiece(token) for token in
                              tokens[start_position: (end_position + 1)]]
                    answer_text = tokenizer.sp_model.DecodePieces(pieces)
                    tf.logging.info("start_position: %d" % (start_position))
                    tf.logging.info("end_position: %d" % (end_position))
                    tf.logging.info(
                        "answer: %s" % (tokenization.printable_text(answer_text)))

                    # note(zhiliny): With multi processing,
                    # the example_index is actually the index within the current process
                    # therefore we use example_index=None to avoid being used in the future.
                    # The current code does not use example_index of training data.
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
                tokens=[tokenizer.sp_model.IdToPiece(x) for x in tokens],
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
            output_fn(feature)

            unique_id += 1
            if span_is_impossible:
                cnt_neg += 1
            else:
                cnt_pos += 1

    tf.logging.info("Total number of instances: {} = pos {} neg {}".format(
        cnt_pos + cnt_neg, cnt_pos, cnt_neg))


def entity_mask(total_len, start_index, max_entity_len, dtype):
    # it seems without this line running on GPU will fail
    # start index will not exceed the boundary, which is ensured during pre-process
    # so this line will not be different
    entity_start_pos = tf.minimum(total_len - 1, start_index)
    # entity_start_pos = start_index # this is the actual behavior i want.
    entity_end_pos = tf.minimum(total_len, start_index + max_entity_len)
    entity_len = entity_end_pos - entity_start_pos

    def fn(elem):
        elem_len, elem_start_pos, elem_end_pos = elem
        mask = tf.ones([elem_len, elem_len], dtype=dtype)
        mask = tf.matrix_band_part(mask, -1, 0)
        elem_mask = tf.pad(mask, paddings=[[elem_start_pos, total_len - elem_end_pos],
                                           [elem_start_pos, total_len - elem_end_pos]])

        return elem_mask

    res_mask = tf.map_fn(fn, [entity_len, entity_start_pos, entity_end_pos], dtype=dtype)

    return res_mask


def create_v2_model(albert_config, is_training, input_ids, input_mask,
                    segment_ids, use_one_hot_embeddings, features,
                    max_seq_length, start_n_top, end_n_top, dropout_prob, syntax_mask,CLS_GELU=False):
    """Creates a classification model."""
    model = modeling.AlbertModel(
        config=albert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        syntax_mask=syntax_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    output = model.get_sequence_output()
    shape = tf.shape(output)
    bsz, seq_length = shape[0], shape[1]
    return_dict = {}


    # B X T
    span_feature = tf.cast(features["mask_start"], dtype=output.dtype)
    # B X T X 1
    span_feature = tf.expand_dims(span_feature, axis=2)
    output = tf.concat([output, span_feature], axis=-1)

    return_dict["output"] = output
    # T X B X D
    output = tf.transpose(output, [1, 0, 2])

    # invalid position mask such as query and special symbols (PAD, SEP, CLS)
    p_mask = tf.cast(features["p_mask"], dtype=tf.float32)

    # logit of the start position
    with tf.variable_scope("start_logits"):
        start_logits = tf.layers.dense(
            output,
            1,
            kernel_initializer=modeling.create_initializer(
                albert_config.initializer_range))
        start_logits = tf.transpose(tf.squeeze(start_logits, -1), [1, 0])
        start_logits_masked = start_logits * p_mask - 1e30 * (1 - p_mask)
        start_log_probs = tf.nn.log_softmax(start_logits_masked, -1)
        start_probs = tf.nn.softmax(start_logits_masked, -1)

    # logit of the end position
    with tf.variable_scope("end_logits"):

        start_top_log_probs, start_top_index = tf.nn.top_k(
            start_log_probs, k=start_n_top)

        # T X B X H
        output_alpha = tf.layers.dense(
            output,
            output.get_shape().as_list()[2],
            kernel_initializer=modeling.create_initializer(
                albert_config.initializer_range),
            name="dense_0")
        # BK X T X T
        en_msk = entity_mask(seq_length, tf.reshape(start_top_index, shape=[-1]), MAX_ANSWER_LENGTH,
                             tf.float32)
        # B X K X T X T
        en_msk = tf.reshape(en_msk, shape=[bsz, -1, seq_length, seq_length])

        # B X T X H
        output_alpha = tf.transpose(output_alpha, [1, 0, 2])
        # B X 1 X T X T
        att = tf.expand_dims(tf.matmul(output_alpha, tf.transpose(output, [1, 2, 0])), axis=1)
        att = att * en_msk

        print("Using dk_div")
        att = att / tf.sqrt(tf.cast(albert_config.hidden_size, att.dtype))
        att = att - ((1.0 - en_msk) * 1e30)
        # B X K X T1 X T2
        att = tf.nn.softmax(att)
        # B X K X T1 X H
        end_logits = tf.einsum("abcd,dae->abce", att, output)


        print("Using residual")
        end_logits = end_logits + tf.expand_dims(tf.transpose(output, [1, 0, 2]), axis=1)

        end_logits = contrib_layers.layer_norm(end_logits, begin_norm_axis=-1)

        # B X K X T1 X 1
        end_logits = tf.layers.dense(
            end_logits,
            1,
            kernel_initializer=modeling.create_initializer(
                albert_config.initializer_range),
            name="dense_1")

        end_logits = tf.squeeze(end_logits, axis=-1)
        end_logits_masked = end_logits * p_mask[:, None] - 1e30 * (1 - p_mask[:, None])
        end_log_probs = tf.nn.log_softmax(end_logits_masked, -1)  # B K 1
        end_probs = tf.nn.softmax(end_logits_masked, -1)
        end_top_log_probs, end_top_index = tf.nn.top_k(
            end_log_probs, k=end_n_top)
        end_top_log_probs = tf.reshape(
            end_top_log_probs,
            [-1, start_n_top * end_n_top])
        end_top_index = tf.reshape(
            end_top_index,
            [-1, start_n_top * end_n_top])

    if is_training:
        return_dict["start_log_probs"] = start_log_probs
        return_dict["end_log_probs"] = end_log_probs
        return_dict["start_probs"] = start_probs
        return_dict["end_probs"] = end_probs
    else:
        return_dict["start_top_log_probs"] = start_top_log_probs
        return_dict["start_top_index"] = start_top_index
        return_dict["end_top_log_probs"] = end_top_log_probs
        return_dict["end_top_index"] = end_top_index

    # an additional layer to predict answerability
    with tf.variable_scope("answer_class"):
        # get the representation of CLS
        cls_feature = model.get_pooled_output()


        ans_feature = tf.layers.dense(
            cls_feature,
            albert_config.hidden_size,
            activation=modeling.get_activation("gelu") if CLS_GELU else tf.tanh,
            kernel_initializer=modeling.create_initializer(
                albert_config.initializer_range),
            name="dense_0")

        ans_feature = tf.layers.dropout(ans_feature, dropout_prob,
                                        training=is_training)



        cls_logits = tf.layers.dense(
            ans_feature,
            2,
            kernel_initializer=modeling.create_initializer(
                albert_config.initializer_range),
            name="dense_1",
            use_bias=False)


        return_dict["cls_logits"] = cls_logits

    return return_dict


def v2_model_fn_builder(albert_config,
                        use_one_hot_embeddings,
                        max_seq_length, start_n_top,
                        end_n_top, dropout_prob, CLS_GELU=False, LOG=False):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        # unique_ids = features["unique_ids"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        outputs = create_v2_model(
            albert_config=albert_config,
            is_training=is_training,
            input_ids=input_ids,
            syntax_mask=None,  # (features['syntax_mask'], FLAGS.SG_Layer) if FLAGS.SG_Layer > 0 else None,
            input_mask=input_mask,
            segment_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings,
            features=features,
            max_seq_length=max_seq_length,
            start_n_top=start_n_top,
            end_n_top=end_n_top,
            dropout_prob=dropout_prob,
            CLS_GELU=CLS_GELU)



        predictions = {
            "unique_ids": features["unique_ids"],
            "start_top_index": outputs["start_top_index"],
            "start_top_log_probs": outputs["start_top_log_probs"],
            "end_top_index": outputs["end_top_index"],
            "end_top_log_probs": outputs["end_top_log_probs"],
            "cls_logits": (tf.nn.log_softmax(outputs["cls_logits"])) if LOG else (tf.nn.softmax(outputs["cls_logits"]) - 0.5),
            "bert_output": outputs['output']
        }
        output_spec = tf.estimator.EstimatorSpec(
            mode=mode, predictions=predictions)


        return output_spec

    return model_fn


def input_fn_builder(input_file, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "unique_ids": tf.FixedLenFeature([], tf.int64),
        "input_ids": tf.VarLenFeature(tf.int64),
        "input_mask": tf.VarLenFeature(tf.int64),
        "segment_ids": tf.VarLenFeature(tf.int64),
        "p_mask": tf.VarLenFeature(tf.int64),
        "mask_start": tf.VarLenFeature(tf.int64),
        "mask_end": tf.VarLenFeature(tf.int64),
    }

    # if FLAGS.SG_Layer > 0:
    #     name_to_features["syntax_mask"] = tf.VarLenFeature(tf.int64)

    if is_training:
        name_to_features["start_positions"] = tf.FixedLenFeature([], tf.int64)
        name_to_features["end_positions"] = tf.FixedLenFeature([], tf.int64)
        name_to_features["is_impossible"] = tf.FixedLenFeature([], tf.int64)

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)

            if isinstance(t, tf.sparse.SparseTensor):
                t = tf.sparse.to_dense(t)

            example[name] = t
        if "syntax_mask" in example:
            data_len = tf.shape(example['input_ids'])[0]
            example["syntax_mask"] = tf.reshape(example["syntax_mask"], shape=[data_len, data_len])

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = BATCH_SIZE
        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)

        input_shape = {
            "unique_ids": tf.TensorShape([]),
            "input_ids": tf.TensorShape([None]),
            "input_mask": tf.TensorShape([None]),
            "segment_ids": tf.TensorShape([None]),
            "p_mask": tf.TensorShape([None]),
            "mask_start": tf.TensorShape([None]),
            "mask_end": tf.TensorShape([None])
        }

        # if FLAGS.SG_Layer > 0:
        #     input_shape["syntax_mask"] = tf.TensorShape([None, None])

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=5000)
            input_shape['start_positions'] = tf.TensorShape([])
            input_shape['end_positions'] = tf.TensorShape([])
            input_shape['is_impossible'] = tf.TensorShape([])

        d = d \
            .map(lambda record: _decode_record(record, name_to_features)) \
            .padded_batch(batch_size=batch_size,
                          padded_shapes=input_shape,
                          drop_remainder=drop_remainder) \
            .prefetch(5000)

        return d

    return input_fn


def accumulate_predictions_v2(result_dict, cls_dict, all_examples,
                              all_features, all_results, n_best_size,
                              max_answer_length, start_n_top, end_n_top):
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

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive

        for (feature_index, feature) in enumerate(features):
            if feature.unique_id not in result_dict[example_index]:
                result_dict[example_index][feature.unique_id] = {}
            result = unique_id_to_result[feature.unique_id]
            cur_null_score = result.cls_logits

            # if we could have irrelevant answers, get the min score of irrelevant
            score_null = min(score_null, cur_null_score)

            doc_offset = feature.tokens.index("[SEP]") + 1
            for i in range(start_n_top):
                for j in range(end_n_top):
                    start_log_prob = result.start_top_log_probs[i]
                    start_index = result.start_top_index[i]

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
        if example_index not in cls_dict:
            cls_dict[example_index] = []
        cls_dict[example_index].append(score_null)

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


def write_predictions_v2(result_dict,  all_examples, all_features,
                         all_results, n_best_size):

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result


    all_nbest_json = collections.OrderedDict()


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
                        null_odds=null_odds / len(logprobs)
                    ))

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_log_prob + x.end_log_prob),
            reverse=True)

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            # if isinstance(null_score_diff_threshold, float) and pred.null_odds >= null_score_diff_threshold:
            #     continue
            feature = features[pred.feature_index]

            tok_start_to_orig_index = feature.tok_start_to_orig_index
            tok_end_to_orig_index = feature.tok_end_to_orig_index
            start_orig_pos = tok_start_to_orig_index[pred.start_index]
            end_orig_pos = tok_end_to_orig_index[pred.end_index]

            paragraph_text = example.paragraph_text
            final_text = paragraph_text[start_orig_pos: end_orig_pos + 1].strip()

            if final_text in seen_predictions:
                continue

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
        # predict null answers when null threshold is provided


        all_nbest_json[example.qas_id] = nbest_json
        assert len(nbest_json) >= 1




    return all_nbest_json


def predict(data_json, model_dir, CLS_GELU=False, LOG=False, config=None, tok=None):
    eval_examples = read_squad_examples(input_data=data_json, is_training=False)

    predict_feature_file = "predict.tf_record.out4"
    predict_feature_left_file = "predict_left.out4"
    if config:
        albert_config = config
    else:
        albert_config = modeling.AlbertConfig.from_json_file("configs/albert/albert_config.json")

    if tok:
        tokenizer = tok
    else:
        tokenizer = tokenization.FullTokenizer(
            vocab_file=None,
            do_lower_case=True,
            spm_model_file="configs/albert/30k-clean.model")

    if tf.gfile.Exists(predict_feature_file) and tf.gfile.Exists(predict_feature_left_file):
        tf.logging.info("Loading eval features from {}".format(predict_feature_left_file))
        with tf.gfile.Open(predict_feature_left_file, "rb") as fin:
            eval_features = pickle.load(fin)
    else:
        eval_writer = FeatureWriter(
            filename=predict_feature_file, is_training=False)
        eval_features = []

        def append_feature(feature):

            fea = InputFeatures(
                tok_start_to_orig_index=feature.tok_start_to_orig_index,
                tok_end_to_orig_index=feature.tok_end_to_orig_index,
                unique_id=feature.unique_id,
                tokens=feature.tokens,
                token_is_max_context=feature.token_is_max_context,
                example_index=feature.example_index
            )

            eval_features.append(fea)
            eval_writer.process_feature(feature)

        convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=512,
            doc_stride=128,
            max_query_length=64,
            is_training=False,
            output_fn=append_feature,
            do_lower_case=True,
            generate_sg=False
        )
        eval_writer.close()

        with tf.gfile.Open(predict_feature_left_file, "wb") as fout:
            pickle.dump(eval_features, fout)

    predict_input_fn = input_fn_builder(
        input_file=predict_feature_file,
        is_training=False,
        drop_remainder=False)

    run_config = tf.estimator.RunConfig(
        model_dir=model_dir)

    model_fn = v2_model_fn_builder(
        albert_config=albert_config,
        use_one_hot_embeddings=False,
        max_seq_length=512,
        start_n_top=START_N,
        end_n_top=END_N,
        dropout_prob=0.1,
        CLS_GELU=CLS_GELU,
        LOG=LOG)
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config
    )

    def get_result(checkpoint):
        all_results = []
        for result in estimator.predict(
                predict_input_fn, yield_single_examples=True,
                checkpoint_path=checkpoint):
            if len(all_results) % 1000 == 0:
                tf.logging.info("Processing example: %d" % (len(all_results)))
            unique_id = int(result["unique_ids"])
            start_top_log_probs = (
                [float(x) for x in result["start_top_log_probs"].flat])
            start_top_index = [int(x) for x in result["start_top_index"].flat]
            end_top_log_probs = (
                [float(x) for x in result["end_top_log_probs"].flat])
            end_top_index = [int(x) for x in result["end_top_index"].flat]
            cls_logits = float(result["cls_logits"].flat[0])
            all_results.append(
                RawResultV2(
                    unique_id=unique_id,
                    start_top_log_probs=start_top_log_probs,
                    start_top_index=start_top_index,
                    end_top_log_probs=end_top_log_probs,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits))


        result_dict = {}
        cls_dict = {}
        accumulate_predictions_v2(
            result_dict, cls_dict, eval_examples, eval_features,
            all_results, N_BEST, MAX_ANSWER_LENGTH,
            START_N, END_N)
        return write_predictions_v2(result_dict,
                                    eval_examples,
                                    eval_features,
                                    all_results,
                                    N_BEST)

    checkpoint_path = os.path.join(model_dir, "model.ckpt-best")
    return get_result(checkpoint_path)