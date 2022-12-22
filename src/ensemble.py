# coding=utf-8

import json
import argparse
import collections
import tensorflow as tf
from models.electra import predict as electra_predict
from models.albert import predict as albert_predict
from utils.postprocess import post_process

OPTS = None


def parse_args():
    parser = argparse.ArgumentParser('submit script')
    parser.add_argument('data_file', metavar='data.json', help='Input data JSON file.')
    parser.add_argument('pred_file', metavar='pred.json', help='Model predictions.')
    return parser.parse_args()


def sort_candidates(candidates,
                    length_penalty=0.02,
                    length_penalty_power=1.00,
                    length_penalty_low_limit=5):
    max_count = -1
    count_key = {}
    for key, score_list in candidates.items():
        count, _ = score_list
        if count not in count_key:
            count_key[count] = []
        count_key[count].append(key)
        if count > max_count:
            max_count = count

    max_score = -10000000
    best_candidate = None

    candidate_pool = []
    candidate_pool.extend(count_key[max_count])

    for cand in candidate_pool:
        penalty_length = max(len(cand.split()), length_penalty_low_limit) - length_penalty_low_limit
        accu_score = candidates[cand][1] - length_penalty * pow(penalty_length, length_penalty_power)
        if accu_score > max_score:
            max_score = accu_score
            best_candidate = cand

    return best_candidate


def normal_average(ori_scores):
    mormal_scores = {}
    for key in ori_scores.keys():
        prob_list = ori_scores[key]
        prob_list.sort(reverse=True)
        turn_off_index = None
        filtered_prob_list = prob_list[:turn_off_index]
        mormal_scores[key] = len(prob_list), sum(filtered_prob_list) / len(filtered_prob_list)
    return mormal_scores


def merge(results, start_weight=0.35):
    merged_result = collections.OrderedDict()
    merged_null_odds = collections.OrderedDict()
    result_count = len(results)
    for id in results[0][0].keys():
        id_cache = {"candidates": {}}
        threshold = 0.0
        for res in results:
            local_threshold = []
            for ans_dict in res[0][id][:res[2]]:
                answer = ans_dict['text']
                probbability = ans_dict['start_log_prob'] * start_weight + ans_dict['end_log_prob'] * (1 - start_weight)
                if answer not in id_cache["candidates"]:
                    id_cache["candidates"][answer] = []
                id_cache["candidates"][answer].append(probbability - ans_dict['null_odds'] * 2)
                local_threshold.append(ans_dict['null_odds'])
            threshold += (sum(local_threshold)/len(local_threshold) - res[1])
        threshold /= result_count

        normal_candidates = normal_average(id_cache["candidates"])
        sorted_answer = sort_candidates(normal_candidates)
        merged_result[id] = sorted_answer
        merged_null_odds[id] = threshold

    return merged_result, merged_null_odds


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.gfile.Open(OPTS.data_file) as predict_file:
        data_json = json.load(predict_file)["data"]

    all_nbest_json_1 = albert_predict(data_json, "C_G3B")
    all_nbest_json_2 = albert_predict(data_json, "C_G7A3", True)
    all_nbest_json_3 = electra_predict(data_json, "WD1")
    all_nbest_json_4 = electra_predict(data_json, "WD4")
    all_nbest_json_5 = albert_predict(data_json, "C_R2_1", True, True, True)
    all_nbest_json_6 = albert_predict(data_json, "C_OUT4_2", False, False, True)

    results = [(all_nbest_json_1, -1.00, 20),
               (all_nbest_json_2, -0.25, 20),
               (all_nbest_json_3, -0.11, 12),
               (all_nbest_json_4, -0.01, 12),
               (all_nbest_json_5, -0.35, 12),
               (all_nbest_json_6, -0.38, 12)]

    merged_preds, merged_null_odds = merge(results)

    predictions = {}
    for qid in merged_preds.keys():
        if merged_null_odds[qid] <= -0.4211052745580673:
            predictions[qid] = merged_preds[qid]
        else:
            predictions[qid] = ""

    predictions = post_process(predictions, data_json)

    with tf.gfile.GFile(OPTS.pred_file, "w") as writer:
        writer.write(json.dumps(predictions, indent=4) + "\n")


if __name__ == "__main__":
    OPTS = parse_args()
    tf.app.run()
