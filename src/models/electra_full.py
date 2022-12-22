# coding=utf-8
import os
import six
import pickle
import collections
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers as contrib_layers
from utils.data import read_squad_examples
from utils.data import FeatureWriter
from utils.data import InputFeatures
from utils.data import RawResultV2
from utils import tokenization
from models import electra_modeling
from utils import squad_utils

BATCH_SIZE = 8
MAX_ANSWER_LENGTH = 30
START_N = 5
END_N = 5
N_BEST = 20


def get_relative_position_embedding_slice(max_length, seq_length,relative_position_embedding, max_relative_length):


    look_up_table = [[[None] for _ in range(max_length)] for _ in range(max_length)]
    count = 0
    for i in range(max_length):
        for j in range(max_length):
            count += 1
            if count % 10000==0:
                tf.logging.info("generating lookup table of %d/%d" % (count, max_length * max_length))
            dis = max(min(i-j + max_relative_length, 2*max_relative_length),0)
            look_up_table[i][j] = [dis]


    slices = tf.gather_nd(relative_position_embedding, look_up_table)
    slices = tf.slice(slices, begin=[0, 0], size=[seq_length, seq_length])

    return slices


def accumulate_predictions_v2(result_dict, all_examples,
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

                    cur_null_score = result.cls_logits #+ (result.start_null_logits + result.end_null_logits[i]) / 2

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
                        (start_log_prob, end_log_prob, cur_null_score, result.start_null_logits,
                         result.end_null_logits[i]))


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
                                 output_fn, do_lower_case, generate_sg=False, use_plausible_answer=False):
    """Loads a data file into a list of `InputBatch`s."""

    cnt_pos, cnt_neg = 0, 0
    unique_id = 1000000000
    max_n, max_m = 1024, 1024
    f = np.zeros((max_n, max_m), dtype=np.float32)

    for (example_index, example) in enumerate(examples):

        if example_index % 100 == 0:
            tf.logging.info("Converting {}/{} pos {} neg {}".format(
                example_index, len(examples), cnt_pos, cnt_neg))

        query_pieces = tokenizer.tokenize(tokenization.preprocess_text(
            example.question_text, lower=do_lower_case))

        para_text = tokenization.preprocess_text(example.paragraph_text, lower=do_lower_case)
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
                    if (tokenization.preprocess_text(
                            example.paragraph_text[i], lower=do_lower_case,
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

        # if generate_sg:
        #     query_text = "".join(query_pieces).replace(
        #         tokenization.SPIECE_UNDERLINE.decode("utf-8"), " ")
        #     syntax_pas = parse_text(tok_cat_text, para_tokens)
        #     syntax_que = parse_text(query_text, query_pieces, text_length=max_query_length)

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

            ##
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
                tf.logging.info("input_pieces: %s" % " ".join(tokenizer.convert_ids_to_tokens(tokens)))
                tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                tf.logging.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                tf.logging.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

                if is_training and span_is_impossible:
                    pieces = [tokenizer.convert_ids_to_tokens([token])[0] for token in
                              tokens[start_position: (end_position + 1)]]
                    answer_text = " ".join(pieces)
                    tf.logging.info("impossible example span")
                    tf.logging.info("start_position: %d" % (start_position))
                    tf.logging.info("end_position: %d" % (end_position))
                    tf.logging.info(
                        "plausible answer: %s" % (tokenization.printable_text(answer_text)))

                if is_training and not span_is_impossible:
                    pieces = [tokenizer.convert_ids_to_tokens([token])[0] for token in
                              tokens[start_position: (end_position + 1)]]
                    answer_text = " ".join(pieces)
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
            output_fn(feature)

            unique_id += 1
            if span_is_impossible:
                cnt_neg += 1
            else:
                cnt_pos += 1

    tf.logging.info("Total number of instances: {} = pos {} neg {}".format(
        cnt_pos + cnt_neg, cnt_pos, cnt_neg))



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
                    start_n_top, end_n_top, syntax_mask,
                    GLU=False,
                    CLS_WITH_START_INFO=True,
                    PAS_ATT=False,
                    QUS_ATT=False,
                    relative_positions=0,
                    ):
    """Creates a classification model."""
    model = electra_modeling.BertModel(
        bert_config=albert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        syntax_mask=syntax_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings,
        scope="electra")

    output = model.get_sequence_output()
    shape = tf.shape(output)
    bsz, seq_length = shape[0], shape[1]
    return_dict = {}

    # B X T


    # invalid position mask such as query and special symbols (PAD, SEP, CLS)
    p_mask = tf.cast(features["p_mask"], dtype=tf.float32)
    with tf.variable_scope("att_layer", reuse=tf.AUTO_REUSE):
        if relative_positions > 0:

            relative_position_embedding = tf.get_variable(
                name="relative_positions",
                shape=[2 * relative_positions + 1],
                dtype=tf.float32,
                trainable=True,
                initializer=electra_modeling.create_initializer(albert_config.initializer_range))
            qus_position_slices = get_relative_position_embedding_slice(
                512, seq_length,
                relative_position_embedding,
                relative_positions)
            pas_position_slices = get_relative_position_embedding_slice(
                512, seq_length,
                relative_position_embedding,
                relative_positions)

        if QUS_ATT:
            tf.logging.info("using qus_att")
            hsz = albert_config.hidden_size
            question_context = []

            with tf.variable_scope("question_attention_%d" % 0, reuse=tf.AUTO_REUSE):
                qus_Q = tf.layers.dense(
                    output,
                    hsz,
                    kernel_initializer=electra_modeling.create_initializer(
                        albert_config.initializer_range),
                    name="qus_Q"
                )
                qus_K = tf.layers.dense(
                    output,
                    hsz,
                    kernel_initializer=electra_modeling.create_initializer(
                        albert_config.initializer_range),
                    name="qus_K"
                )
                qus_V = tf.layers.dense(
                    output,
                    hsz,
                    kernel_initializer=electra_modeling.create_initializer(
                        albert_config.initializer_range),
                    name="qus_V"
                )

                question_mask = tf.expand_dims(1 - p_mask, axis=1)
                qus_att = tf.matmul(qus_Q, tf.transpose(qus_K, [0, 2, 1]))
                if relative_positions > 0:
                    qus_att += qus_position_slices
                qus_att = qus_att * question_mask
                qus_att = qus_att / tf.sqrt(tf.cast(hsz, output.dtype))

                qus_att = qus_att - ((1 - question_mask) * 1e30)
                tf.logging.info("using question mask")
                qus_att = tf.nn.softmax(qus_att)
                question_context.append(tf.matmul(qus_att, qus_V))
            question_context = tf.concat(question_context, axis=-1)
            output = question_context + output

        if PAS_ATT:
            tf.logging.info("using pas_att")
            hsz = albert_config.hidden_size
            pas_context = []
            with tf.variable_scope("passage_attention_%d" % 0, reuse=tf.AUTO_REUSE):
                pas_Q = tf.layers.dense(
                    output,
                    hsz,
                    kernel_initializer=electra_modeling.create_initializer(
                        albert_config.initializer_range),
                    name="pas_Q"
                )
                pas_K = tf.layers.dense(
                    output,
                    hsz,
                    kernel_initializer=electra_modeling.create_initializer(
                        albert_config.initializer_range),
                    name="pas_K"
                )
                pas_V = tf.layers.dense(
                    output,
                    hsz,
                    kernel_initializer=electra_modeling.create_initializer(
                        albert_config.initializer_range),
                    name="pas_V"
                )
                pas_mask = tf.expand_dims(p_mask, axis=1)
                pas_att = tf.matmul(pas_Q, tf.transpose(pas_K, [0, 2, 1]))
                if relative_positions > 0:
                    pas_att += pas_position_slices
                pas_att = pas_att * pas_mask
                pas_att = pas_att / tf.sqrt(tf.cast(hsz, output.dtype))

                pas_att = pas_att - ((1 - pas_mask) * 1e30)
                tf.logging.info("using pas mask")
                pas_att = tf.nn.softmax(pas_att)
                pas_context.append(tf.matmul(pas_att, pas_V))

            pas_context = tf.concat(pas_context, axis=-1)
            output = output + pas_context
        output = contrib_layers.layer_norm(output, begin_norm_axis=-1)

    span_feature = tf.cast(features["mask_start"], dtype=output.dtype)
    # B X T X 1
    span_feature = tf.expand_dims(span_feature, axis=2)
    output = tf.concat([output, span_feature], axis=-1)
    output_shape = output.get_shape().as_list()

    # logit of the start position
    with tf.variable_scope("start_logits"):
        start_logits = tf.layers.dense(
            output,
            1,
            kernel_initializer=electra_modeling.create_initializer(
                albert_config.initializer_range))
        start_logits = tf.squeeze(start_logits, -1)
        start_logits_masked = start_logits * p_mask - 1e30 * (1 - p_mask)
        start_log_probs = tf.nn.log_softmax(start_logits_masked, -1)


    # logit of the end position
    with tf.variable_scope("end_logits"):
        # during inference, compute the end logits based on beam search
        # B X K
        start_top_log_probs, start_top_index = tf.nn.top_k(
            start_log_probs, k=start_n_top)
        en_msk = entity_mask(seq_length, tf.reshape(start_top_index, shape=[-1]), 30,
                             tf.float32)
        # B X K X T X T
        en_msk = tf.reshape(en_msk, shape=[bsz, -1, seq_length, seq_length])

        end_logits = []
        hidden_size = output_shape[2]

        with tf.variable_scope("entity_mask_%d" % 0, reuse=tf.AUTO_REUSE):

            output_Q = tf.layers.dense(
                output,
                hidden_size,
                kernel_initializer=electra_modeling.create_initializer(
                    albert_config.initializer_range),
                name="dense_q")

            output_K = tf.layers.dense(
                output,
                hidden_size,
                kernel_initializer=electra_modeling.create_initializer(
                    albert_config.initializer_range),
                name="dense_k")
            output_V = tf.layers.dense(
                output,
                hidden_size,
                kernel_initializer=electra_modeling.create_initializer(
                    albert_config.initializer_range),
                name="dense_v"
            )
            att = tf.expand_dims(tf.matmul(output_Q, tf.transpose(output_K, [0, 2, 1])), axis=1)
            att = att * en_msk
            att = att / tf.sqrt(tf.cast(hidden_size, att.dtype))
            att = att - ((1.0 - en_msk) * 1e30)
            # B X K X T1 X T2
            att = tf.nn.softmax(att)
            end_logits.append(tf.einsum("abcd,ade->abce", att, output_V))

        end_logits = tf.concat(end_logits, -1)
        end_logits = end_logits + tf.expand_dims(output, axis=1)
        end_logits = contrib_layers.layer_norm(end_logits, begin_norm_axis=-1)

        # B X K X T1 X 1
        end_logits = tf.layers.dense(
            end_logits,
            1,
            kernel_initializer=electra_modeling.create_initializer(
                albert_config.initializer_range),
            name="dense_3")

        end_logits = tf.squeeze(end_logits, axis=-1)
        ex_p_mask = p_mask[:, None]
        end_logits_masked = end_logits * ex_p_mask - 1e30 * (1 - ex_p_mask)
        end_log_probs = tf.nn.log_softmax(end_logits_masked, -1)  # B K 1

        end_top_log_probs, end_top_index = tf.nn.top_k(end_log_probs, k=end_n_top)
        end_top_log_probs = tf.reshape(
            end_top_log_probs,
            [-1, start_n_top * end_n_top])
        end_top_index = tf.reshape(
            end_top_index,
            [-1, start_n_top * end_n_top])

    return_dict["start_top_log_probs"] = start_top_log_probs
    return_dict["start_top_index"] = start_top_index
    return_dict["end_top_log_probs"] = end_top_log_probs
    return_dict["end_top_index"] = end_top_index
    return_dict["start_null_logits"] = start_log_probs[:, 0]
    return_dict["end_null_logits"] = end_log_probs[:, :, 0]

    # an additional layer to predict answerability
    with tf.variable_scope("answer_class"):
        # get the representation of CLS
        cls_feature = output[:, 0, :]

        # get the representation of START
        if CLS_WITH_START_INFO:
            # get the representation of START
            start_p = tf.nn.softmax(start_logits_masked, axis=-1,
                                    name="softmax_start")
            start_feature = tf.einsum("blh,bl->bh", output, start_p)
            ans_feature = tf.concat([start_feature, cls_feature], -1)
        else:
            ans_feature = cls_feature

        if GLU:
            ans_feature = tf.layers.dense(
                ans_feature,
                albert_config.hidden_size * 2 // 3,
                activation=tf.tanh,
                kernel_initializer=electra_modeling.create_initializer(
                    albert_config.initializer_range),
                name="dense_0")
            ans_feature = ans_feature * tf.layers.dense(
                ans_feature,
                albert_config.hidden_size * 2 // 3,
                kernel_initializer=electra_modeling.create_initializer(
                    albert_config.initializer_range),
                name="dense_glu"
            )
        else:
            ans_feature = tf.layers.dense(
                ans_feature,
                albert_config.hidden_size,
                activation=tf.tanh,
                kernel_initializer=electra_modeling.create_initializer(
                    albert_config.initializer_range),
                name="dense_0")

        cls_logits = tf.layers.dense(
            ans_feature,
            1,
            kernel_initializer=electra_modeling.create_initializer(
                albert_config.initializer_range),
            name="dense_1",
            use_bias=False)
        cls_logits = tf.squeeze(cls_logits, -1)
        return_dict["cls_logits"] = cls_logits

    return return_dict


def v2_model_fn_builder(albert_config, use_one_hot_embeddings, start_n_top,
                        end_n_top,
                        GLU=False,
                        CLS_WITH_START_INFO=True,
                        PAS_ATT=False,
                        QUS_ATT=False,
                        relative_positions=0
                        ):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

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
            start_n_top=start_n_top,
            end_n_top=end_n_top,
            GLU=GLU,
            CLS_WITH_START_INFO=CLS_WITH_START_INFO,
            PAS_ATT=PAS_ATT,
            QUS_ATT=QUS_ATT,
            relative_positions=relative_positions
        )

        predictions = {
            "unique_ids": features["unique_ids"],
            "start_top_index": outputs["start_top_index"],
            "start_top_log_probs": outputs["start_top_log_probs"],
            "end_top_index": outputs["end_top_index"],
            "end_top_log_probs": outputs["end_top_log_probs"],
            "cls_logits": tf.math.log_sigmoid(outputs["cls_logits"]),
            "start_null_logits": outputs["start_null_logits"],
            "end_null_logits": outputs["end_null_logits"]
        }
        output_spec = tf.estimator.EstimatorSpec(
            mode=mode, predictions=predictions)

        return output_spec

    return model_fn


def predict(data_json, model_dir,
            GLU=False, CLS_WITH_START_INFO=True,
            PAS_ATT=False, QUS_ATT=False, config=None, tok=None, relative_positions=0):
    eval_examples = read_squad_examples(input_data=data_json, is_training=False)

    predict_feature_file = "predict.tf_record.electra"
    predict_feature_left_file = "predict_left.electra"
    if tok:
        tokenizer = tok
    else:
        tokenizer = tokenization.FullTokenizer(
            vocab_file="configs/electra/vocab.txt",
            do_lower_case=True)

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
        model_dir=model_dir,
    )

    if config:
        cfg = config
    else:
        cfg = electra_modeling.BertConfig.from_json_file("configs/electra/config.json")

    model_fn = v2_model_fn_builder(
        albert_config=cfg,
        use_one_hot_embeddings=False,
        start_n_top=START_N,
        end_n_top=END_N,
        GLU=GLU,
        CLS_WITH_START_INFO=CLS_WITH_START_INFO,
        PAS_ATT=PAS_ATT,
        QUS_ATT=QUS_ATT,
        relative_positions=relative_positions)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config
    )

    def get_result(checkpoint):
        """Evaluate the checkpoint on SQuAD v2.0."""
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
            start_null_logits = float(result["start_null_logits"].flat[0])
            end_null_logits = [float(x) for x in result["end_null_logits"].flat]

            all_results.append(
                RawResultV2(
                    unique_id=unique_id,
                    start_top_log_probs=start_top_log_probs,
                    start_top_index=start_top_index,
                    end_top_log_probs=end_top_log_probs,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits,  # (start_null_logits + min(end_null_logits)) / 2
                    start_null_logits=start_null_logits,
                    end_null_logits=end_null_logits
                ))

        result_dict = {}
        accumulate_predictions_v2(
            result_dict, eval_examples, eval_features,
            all_results, MAX_ANSWER_LENGTH, START_N, END_N, reverse=False)

        return squad_utils.write_predictions_v2(result_dict,
                                                eval_examples,
                                                eval_features,
                                                all_results,
                                                N_BEST)

    checkpoint_path = os.path.join(model_dir, "model.ckpt-best")
    return get_result(checkpoint_path)



