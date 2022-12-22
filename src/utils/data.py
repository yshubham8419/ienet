import json
import collections
import tensorflow as tf
from utils import tokenization

RawResultV2 = collections.namedtuple(
    "RawResultV2",
    ["unique_id", "start_top_log_probs", "start_top_index",
     "end_top_log_probs", "end_top_index", "cls_logits",
     "start_null_logits",
     "end_null_logits"])


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

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
        s += ", question_text: %s" % (
            tokenization.printable_text(self.question_text))
        s += ", paragraph_text: [%s]" % (" ".join(self.paragraph_text))
        if self.start_position:
            s += ", start_position: %d" % self.start_position
        if self.start_position:
            s += ", end_position: %d" % self.end_position
        if self.start_position:
            s += ", is_impossible: %r" % self.is_impossible
        return s


def read_squad_examples(input_data, is_training):
    """Read a SQuAD json file into a list of SquadExample."""
    examples = []
    for entry in input_data:
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
                    else:
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

    return examples


def create_int_feature(values):
    feature = tf.train.Feature(
        int64_list=tf.train.Int64List(value=list(values)))
    return feature


class FeatureWriter(object):
    """Writes InputFeature to TF example file."""

    def __init__(self, filename, is_training):
        self.filename = filename
        self.is_training = is_training
        self.num_features = 0
        self._writer = tf.python_io.TFRecordWriter(filename)

    def process_feature(self, feature):
        """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
        self.num_features += 1

        features = collections.OrderedDict()
        features["unique_ids"] = create_int_feature([feature.unique_id])
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["p_mask"] = create_int_feature(feature.p_mask)
        features["mask_start"] = create_int_feature(feature.mask_start)
        features["mask_end"] = create_int_feature(feature.mask_end)

        if len(feature.syntax_mask) > 0:
            features["syntax_mask"] = create_int_feature(feature.syntax_mask)

        if self.is_training:
            features["start_positions"] = create_int_feature([feature.start_position])
            features["end_positions"] = create_int_feature([feature.end_position])
            impossible = 0
            if feature.is_impossible:
                impossible = 1
            features["is_impossible"] = create_int_feature([impossible])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        self._writer.write(tf_example.SerializeToString())

    def close(self):
        self._writer.close()
