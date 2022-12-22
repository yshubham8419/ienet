# coding=utf-8
import os
import json
import collections
import argparse
import tensorflow as tf
import utils.tokenization as tokenization
from models.albert import predict as albert_predict
from models.electra import predict as electra_predict
from models.electra_wo_pas_att import predict as electra_wo_pas_att
from models.electra_reverse import predict as electra_reverse_predict
from models.deberta import predict as deberta_predict
from models.albert_modeling import AlbertConfig
from models.electra_modeling import BertConfig
from models.electra_full import predict as electra_full
from postprocess import post_process


id_to_answerable = {}


def parse_args():
    parser = argparse.ArgumentParser('submit script')
    parser.add_argument('data_file', metavar='data.json', help='Input data JSON file.')
    parser.add_argument('pred_file', metavar='pred.json', help='Model predictions.')

    return parser.parse_args()


def load_from_dir(dir_name):
    with open(os.path.join(dir_name, "nbest_predictions.json")) as nbest_f:
        nbest_json = json.load(nbest_f)
        return nbest_json


def sort_candidates(candidates,
                    length_penalty=0.055, # 0.04
                    length_penalty_power=1.2, # 1.0
                    length_penalty_low_limit=12): # 9
    max_count = -1
    count_key = {}
    for key, score_list in candidates.items():
        count = len(score_list[0])
        if count not in count_key:
            count_key[count] = []
        count_key[count].append(key)
        if count > max_count:
            max_count = count

    max_score = -10000000
    best_candidate = None

    candidate_pool = []
    candidate_pool.extend(count_key[max_count]) # 找到投票次数最多的
    # 投票次数最多的里面找到分数最高的
    for cand in candidate_pool:
        penalty_length = max(len(cand.split()), length_penalty_low_limit) - length_penalty_low_limit
        scores = candidates[cand][0]
        accu_score = sum(scores) / len(scores) - length_penalty * pow(penalty_length, length_penalty_power)
        if accu_score > max_score:
            max_score = accu_score
            best_candidate = cand

    return best_candidate


def merge(results, id_2_query, start_weight=0.383):
    merged_result = collections.OrderedDict()
    merged_null_odds = collections.OrderedDict()

    for id in results[0][0].keys():
        id_cache = {"candidates": {}}

        question = id_2_query[id]
        question_tokens = set()

        stop_words = {"a", "an", "in", "at", "the",
                      "it", "with", "and", "or", "as",
                      "to", "if", "on", "about", "that", "be"}

        for tkn in question.split():
            tkn = tkn.lower()
            if tkn not in stop_words:
                question_tokens.add(tkn)

        for res in results:
            for ans_dict in res[0][id][:res[2]]:
                answer = ans_dict['text']
                answer_tokens = [tkn.lower() for tkn in answer.split()]
                probbability = ans_dict['start_log_prob'] * start_weight + ans_dict['end_log_prob'] * (1 - start_weight)

                is_overlap = False
                for tkn in answer_tokens:
                    if tkn in question_tokens:
                        is_overlap = True
                        break

                if is_overlap:
                    probbability -= 0.11

                if answer not in id_cache["candidates"]:
                    id_cache["candidates"][answer] = [[], []]

                null_odds = ans_dict['null_odds']
                gobal_null_odds = res[1]

                if res[3]:
                    null_odds /= 2.0
                    gobal_null_odds /= 2.0

                # if res[4]:
                #     probbability /= 1.0
                #     null_odds /= 1.5
                #     gobal_null_odds *= 0.0

                id_cache["candidates"][answer][1].append(null_odds - 0.0 * gobal_null_odds)
                id_cache["candidates"][answer][0].append(1.0 * probbability - null_odds * 1.0 + 0.0 * gobal_null_odds)

        sorted_answer = sort_candidates(id_cache["candidates"])
        merged_result[id] = sorted_answer
        null_scores = id_cache["candidates"][sorted_answer][1]
        merged_null_odds[id] = sum(null_scores) / len(null_scores)


        if len(null_scores) == len(results):
            merged_null_odds[id] = sum(null_scores) / len(null_scores)
        elif (len(results) - len(null_scores)) == 1:
            merged_null_odds[id] = sum(null_scores) / len(null_scores) + 0.1  # 1.0
        elif (len(results) - len(null_scores)) == 2:
            merged_null_odds[id] = sum(null_scores) / len(null_scores) + 0.1  # 1.2
        else:  # (len(results) - len(null_scores)) == 3:
            merged_null_odds[id] = sum(null_scores) / len(null_scores) + 0.6  # 1.8


    return merged_result, merged_null_odds


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    OPTS = parse_args()

    with open(OPTS.data_file) as f:
        dev_data = json.load(f)['data']
    electra_config = BertConfig.from_json_file("configs/electra/config.json")
    albert_config = AlbertConfig.from_json_file("configs/albert/albert_config.json")

    electra_tokenizer = tokenization.FullTokenizer(
        vocab_file="configs/electra/vocab.txt",
        do_lower_case=True)

    alert_tokenizer = tokenization.FullTokenizer(
        vocab_file=None,
        do_lower_case=True,
        spm_model_file="configs/albert/30k-clean.model")

    all_nbest_json_7 = deberta_predict(dev_data,
                                       "deberta_xlarge_v1_j_6",
                                       "deberta_xlarge_v1_j_6/config.json",
                                       384)

    all_nbest_json_1 = electra_full(dev_data,"large_lr_6", True,
                                    True, True, True, electra_config, electra_tokenizer, 0)
    all_nbest_json_2 = electra_full(dev_data,"large_relative_1", True,
                                    True, True, True, electra_config, electra_tokenizer, 64)
    all_nbest_json_3 = electra_full(dev_data,"large_relative_3", True,
                                    True, True, True, electra_config, electra_tokenizer, 128)

    all_nbest_json_4 = albert_predict(dev_data,
                                      "xxlarge_bsz_12",
                                      CLS_GELU=False,
                                      REVERSE_SEGMENT=True,
                                      USE_V2=False,
                                      null_odds_with_s0e0=False,
                                      PAS_ATT=False,
                                      QUS_ATT=False,
                                      nums_entity_heads=-1,
                                      tok=alert_tokenizer,
                                      config=albert_config)
    all_nbest_json_5 = albert_predict(dev_data,
                                      "xxlarge_v2_1",
                                      CLS_GELU=False,
                                      REVERSE_SEGMENT=True,
                                      USE_V2=False,
                                      null_odds_with_s0e0=False,
                                      PAS_ATT=False,
                                      QUS_ATT=False,
                                      nums_entity_heads=-1,
                                      tok=alert_tokenizer,
                                      config=albert_config                                      )
    all_nbest_json_6 = albert_predict(dev_data, "xxlarge_swd_6",
                                      CLS_GELU=False,
                                      REVERSE_SEGMENT=True,
                                      USE_V2=False,
                                      null_odds_with_s0e0=False,
                                      tok=alert_tokenizer,
                                      config=albert_config)



    # debug
    # all_nbest_json_1 = load_from_dir("large_lr_6")
    # all_nbest_json_2 = load_from_dir("large_relative_1")
    # all_nbest_json_3 = load_from_dir("large_relative_3")
    # all_nbest_json_4 = load_from_dir("xxlarge_bsz_12")
    # all_nbest_json_5 = load_from_dir("xxlarge_v2_1")
    # all_nbest_json_6 = load_from_dir("xxlarge_swd_6")
    # all_nbest_json_7 = load_from_dir("deberta_xlarge_v1_j_6")
    id_2_query = {}

    for entry in dev_data:
        for paragraph in entry["paragraphs"]:
            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                id_2_query[qas_id] = question_text

    out_path = OPTS.pred_file

    results = [
        (all_nbest_json_1, -1.849, 20, False, False), #large_nq_12
        (all_nbest_json_2, -1.741, 20, False, False),
        (all_nbest_json_3, -1.540, 8, False, False),
        (all_nbest_json_4, -1.99, 10, False, False),
        (all_nbest_json_5, -2.41, 20, False, False),
        (all_nbest_json_6, -2.864, 10, False, False),
        (all_nbest_json_7, -3.436, 20, False, False)
    ]

    # for model_settings in in_models:
    #     model_dir, threshold, nbest = model_settings[:3]
    #     if os.path.isdir(model_dir):
    #         nbest_json = load_from_dir(model_dir)
    #         results.append((nbest_json, threshold, nbest, model_settings[3] if len(model_settings) >= 4 else False))

    merged_preds, merged_null_odds = merge(results, id_2_query)
    # verifier_json = verifier_predict(dev_data, id_to_answerable, verifier_dir)
    predictions = {}
    for qid in merged_preds.keys():
        # 这个数是在dev集上搜索得到的阈值
        if merged_null_odds[qid] <= -2.1413820130484447:
            predictions[qid] = merged_preds[qid]
        else:
            predictions[qid] = ""
    predictions = post_process(predictions, dev_data)
    with tf.gfile.GFile(out_path, "w") as writer:
        writer.write(json.dumps(predictions, indent=4) + "\n")

