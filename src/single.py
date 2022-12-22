# coding=utf-8
import collections
import os
import json
import argparse
import tensorflow as tf
from models.electra import predict as electra_predict
from models.albert import predict as albert_predict
from models.electra_spf import predict as electra_spf
from models.electra_wo_pas_att import predict as electra_wo_pas_att
from models.electra_reverse import predict as electra_reverse
from models.electra_full import predict as electra_full
from models.deberta import predict as deberta_predict
from utils.data import InputFeatures

OPTS = None
models = {
    "1": ("r3f_1", electra_predict, False, 0, False, False),
    "2": ("large_all_1", electra_predict, True, 3, False, True),
    "3": ("large_bsz_1", electra_predict, False, 1, False, True),
    "4": ("xxlarge_swd_6", albert_predict, False, False, False, -1),
    "5": ("xxlarge_base_v2_2", albert_predict, True),
    "6": ("/tf_group/xiaotianxiong/R3F_FULL_ENTITY/large_bsz_1_avg", electra_predict, False, 1, False, True),
    "7": ("xxlarge_full_4", albert_predict, False, True, True, -1),
    "8": ("large_full_3", electra_spf, True, True, True, True),
    "9": ("large_nq_12", electra_wo_pas_att, True, True, False, True),
    # "10": ("deberta_xlarge_v1_j_2", deberta_predict, "deberta_xlarge_v1_j_2/config.json", 384),
    "11":("xxlarge_bsz_12", albert_predict, False, False, False, -1),
    "12":("xxlarge_nq_2", albert_predict, False, False, False, -1),
    "13":("xxlarge_v2_1", albert_predict, False, False, False, -1),
    "14":("large_reverse_f_1", electra_reverse, True, True, True, True),
    "15":("large_lr_6", electra_full, True, True, True, True, 0),
    "16":("large_relative_1", electra_full, True, True, True, True, 64),
    "17":("large_relative_3", electra_full, True, True, True, True, 128),
    "18":("deberta_xlarge_v1_j_6", deberta_predict, "deberta_xlarge_v1_j_6/config.json", 384),
}
# electra_reverse: GLU=False, CLS_WITH_START_INFO=True, PAS_ATT=False, QUS_ATT=False,
# electra_full: GLU=False, CLS_WITH_START_INFO=True, PAS_ATT=False, QUS_ATT=False, relative_positions
# electra: settings[2]:qus_att_full settings[3]:entity_mask_version setting[4]:GLU setting[5]:CLS WIH START INFO
# albert: settings[2]: null_odds_with_s0e0 settings[3]:PAS_ATT settings[4]:QUS_ATT settings[5]:nums_entity_heads
# electra_spf: settings[2]: GLU=False, settings[3]: CLS_WITH_START_INFO=True, setting[4]:PAS_ATT=False, setting[5]:QUS_ATT=False

def parse_args():
    parser = argparse.ArgumentParser('submit script')
    parser.add_argument('data_file', metavar='data.json', help='Input data JSON file.')
    parser.add_argument('pred_file', metavar='pred.json', help='Model predictions.')
    parser.add_argument('model_num', metavar='1', help='Input data JSON file.')
    return parser.parse_args()


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.gfile.Open(OPTS.data_file) as predict_file:
        data_json = json.load(predict_file)["data"]

    settings = models[OPTS.model_num]

    if settings[1] == albert_predict:
        def fn():
            return albert_predict(data_json,
                                  settings[0],
                                  CLS_GELU=False,
                                  REVERSE_SEGMENT=True,
                                  USE_V2=False,
                                  null_odds_with_s0e0=settings[2],
                                  PAS_ATT=settings[3],
                                  QUS_ATT=settings[4],
                                  nums_entity_heads=settings[5])
    elif settings[1] == electra_predict:
        def fn():
            return electra_predict(data_json, settings[0], settings[2], settings[3], settings[4], settings[5])
    elif settings[1] == electra_spf:
        def fn():
            return electra_spf(data_json, settings[0], settings[2], settings[3], settings[4], settings[5])
    elif settings[1] == electra_wo_pas_att:
        def fn():
            return electra_wo_pas_att(data_json, settings[0], settings[2], settings[3], settings[4], settings[5])
    elif settings[1] == electra_reverse:
        def fn():
            return electra_reverse(data_json, settings[0], settings[2], settings[3], settings[4], settings[5])

    elif settings[1] == electra_full:
        def fn():
            return electra_full(data_json,
                                settings[0],
                                settings[2],
                                settings[3],
                                settings[4],
                                settings[5],
                                None,
                                None,
                                settings[6])
    elif settings[1] == deberta_predict:
        def fn():
            return deberta_predict(data_json,
                                   settings[0],
                                   settings[2],
                                   settings[3])



    # else:
    #     def sort_candidates(candidates,
    #                         length_penalty=0.02,
    #                         length_penalty_power=1.0,
    #                         length_penalty_low_limit=5):
    #         max_count = -1
    #         count_key = {}
    #         for key, score_list in candidates.items():
    #             count = len(score_list[0])
    #             if count not in count_key:
    #                 count_key[count] = []
    #             count_key[count].append(key)
    #             if count > max_count:
    #                 max_count = count
    #
    #         max_score = -10000000
    #         best_candidate = None
    #
    #         candidate_pool = []
    #         candidate_pool.extend(count_key[max_count])
    #
    #         for cand in candidate_pool:
    #             penalty_length = max(len(cand.split()), length_penalty_low_limit) - length_penalty_low_limit
    #             scores = candidates[cand][0]
    #             accu_score = sum(scores) / len(scores) - length_penalty * pow(penalty_length, length_penalty_power)
    #             if accu_score > max_score:
    #                 max_score = accu_score
    #                 best_candidate = cand
    #
    #         return best_candidate
    #
    #     def load_from_dir(dir_name):
    #         with open(os.path.join(dir_name, "nbest_predictions.json")) as nbest_f:
    #             nbest_json = json.load(nbest_f)
    #             return nbest_json
    #     results =[("r3f_1", -1.95, 20),
    #     ("r3f_2", -1.92, 20),
    #     ("xl_1", -6.38, 20, True),
    #     ("out4", -0.38, 12)]
    #     for model_settings in results:
    #         model_dir, threshold, nbest = model_settings[:3]
    #
    #         if os.path.isdir(model_dir):
    #             nbest_json = load_from_dir(model_dir)
    #             results.append((nbest_json, threshold, nbest, model_settings[3] if len(model_settings) >= 4 else False))
    #     id_to_answerable = {}
    #     def merge(results, id_2_query, start_weight=0.35):
    #         merged_result = collections.OrderedDict()
    #         merged_null_odds = collections.OrderedDict()
    #
    #         for id in results[0][0].keys():
    #             id_cache = {"candidates": {}}
    #
    #             question = id_2_query[id]
    #             question_tokens = set()
    #
    #             for tkn in question.split():
    #                 tkn = tkn.lower()
    #                 if tkn not in {"a", "an", "in", "at"}:
    #                     question_tokens.add(tkn)
    #
    #             for res in results:
    #                 for ans_dict in res[0][id][:res[2]]:
    #                     answer = ans_dict['text']
    #                     answer_tokens = [tkn.lower() for tkn in answer.split()]
    #                     probbability = ans_dict['start_log_prob'] * start_weight + ans_dict['end_log_prob'] * (
    #                                 1 - start_weight)
    #
    #                     is_overlap = False
    #                     for tkn in answer_tokens:
    #                         if tkn in question_tokens:
    #                             is_overlap = True
    #                             break
    #
    #                     if is_overlap:
    #                         probbability -= 0.2
    #
    #                     if answer not in id_cache["candidates"]:
    #                         id_cache["candidates"][answer] = [[], []]
    #
    #                     null_odds = ans_dict['null_odds']
    #                     gobal_null_odds = res[1]
    #
    #                     if res[3]:
    #                         null_odds /= 2
    #                         gobal_null_odds /= 2
    #                     id_cache["candidates"][answer][1].append(null_odds)
    #                     # else:
    #
    #                     id_cache["candidates"][answer][0].append(probbability - null_odds * 3.0 + 6. * gobal_null_odds)
    #
    #             sorted_answer = sort_candidates(id_cache["candidates"])
    #             merged_result[id] = sorted_answer
    #             null_scores = id_cache["candidates"][sorted_answer][1]
    #
    #             if len(null_scores) == len(results):
    #                 merged_null_odds[id] = sum(null_scores) / len(null_scores)
    #                 id_to_answerable[id] = null_scores
    #             else:
    #                 merged_null_odds[id] = +1000.
    #
    #         return merged_result, merged_null_odds
    #
    #     id_2_query = {}
    #
    #     for entry in data_json:
    #         for paragraph in entry["paragraphs"]:
    #             for qa in paragraph["qas"]:
    #                 qas_id = qa["id"]
    #                 question_text = qa["question"]
    #                 id_2_query[qas_id] = question_text
    #     merge(results, id_2_query)
    #
    #     def fn():
    #         return verifier_predict(data_json, id_to_answerable, settings[0])
    all_nbest_json = fn()

    output_nbest_file = os.path.join(
        settings[0], "nbest_predictions.json")

    with tf.gfile.GFile(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")


if __name__ == "__main__":
    OPTS = parse_args()
    tf.app.run()
