import re
import string
import tensorflow as tf
import json
import collections


def normalize_answer(s):
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


def process_impossible_rule(question, ori_answer):
    clean_question = normalize_answer(question)
    clean_ans = normalize_answer(ori_answer)
    clean_question_word = clean_question.split()
    normal_question_word = ['can', 'did', 'do', 'does', 'has', 'could']
    if clean_question_word[0] in normal_question_word:
        if 'or' not in clean_question_word:

            if 'not' in clean_ans:
                return 'not', True
            elif 'no' in clean_ans:
                return 'no', True
            else:
                return '', True

    return ori_answer, False


def rule_filter_ok(original):
    # lower_ori = original.lower()
    # prefixs = [
    #     "because of",
    #     "because",
    #     "by means of ",
    #     "by "
    # ]
    # for key in prefixs:
    #     if lower_ori.startswith(key):
    #         original = original[len(key):].strip()
    #         break

    return original


def remove_bracket(inp_json, dat_json):
    output_json = collections.OrderedDict()
    example_data = dat_json
    count = 0
    for entry in example_data:
        for paragraph in entry["paragraphs"]:
            para_text = paragraph["context"]
            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                pred_answer = inp_json[qas_id]
                output_json[qas_id] = pred_answer
                if not pred_answer:
                    continue
                answer_index = para_text.find(pred_answer)
                if answer_index + len(pred_answer) == len(para_text):
                    continue
                bracket_index = pred_answer.find("(")
                last_bracket_index = pred_answer.find(")")
                if bracket_index > 1 and len(pred_answer) == last_bracket_index + 1:
                    if pred_answer[bracket_index - 1] == ' ' and len(pred_answer[:bracket_index].split()) >= len(
                            pred_answer[bracket_index + 1:].split()):
                        pred_answer = pred_answer[:bracket_index].strip()
                        count += 1

                output_json[qas_id] = rule_filter_ok(pred_answer)
    return output_json


def post_process(inp_json, dat_json):
    output_json = collections.OrderedDict()
    postprocess_count = 0
    remove_bracket_count = 0
    for entry in dat_json:
        for paragraph in entry["paragraphs"]:
            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                pred_answer = inp_json[qas_id]
                output_json[qas_id] = pred_answer
                if not pred_answer:
                    continue
                pred_answer, is_processed = process_impossible_rule(question_text, pred_answer)
                if is_processed:
                    postprocess_count += 1

                output_json[qas_id] = pred_answer

    return remove_bracket(output_json, dat_json)
    # return output_json


if __name__ == "__main__":
    path_inp = "ensemble_predictions.json"
    path_out = "predictions_final.json"
    path_dat = "squad2/dev-v2.0.json"

    with tf.gfile.Open(path_inp, "r") as pred:
        json_inp = json.load(pred)

    with tf.gfile.Open(path_dat, "r") as pred:
        json_dat = json.load(pred)['data']

    json_out = post_process(json_inp, json_dat)

    with tf.gfile.GFile(path_out, "w") as writer:
        writer.write(json.dumps(json_out, indent=4) + "\n")
