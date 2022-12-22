import torch
import logging
import os
import json
import collections
from tqdm import tqdm
from utils import deberta_squad_utils as squad_utils
from transformers import DebertaConfig, DebertaTokenizer
from models.deberta_modeling import DebertaForQAJointly
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RawResultV2J = collections.namedtuple(
    "RawResultV2",
    ["unique_id", "start_top_log_probs", "start_top_index",
     "end_top_log_probs", "end_top_index", "cls_logits"])

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def predict(data_json, model_dir, config_name, max_seq_length):

    eval_feature_file = "predict.deberta"
    eval_batch_size = 4
    tokenizer = DebertaTokenizer.from_pretrained(
        model_dir, do_lower_case=False
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = DebertaConfig.from_json_file(config_name)
    model = DebertaForQAJointly.from_pretrained(model_dir, from_tf=False, config=config,
                                                cls_drop_prob=0.0,
                                                entropy_beta=0.0)
    model.to(device)
    squad_examples = squad_utils.read_squad_examples(data_json, is_training=False)
    if os.path.exists(eval_feature_file):
        logger.info("Loading eval features from cached file %s", eval_feature_file)
        eval_features = torch.load(eval_feature_file)
    else:
        logger.info("Creating features")
        eval_features = squad_utils.convert_examples_to_features(squad_examples,
                                                                 tokenizer=tokenizer,
                                                                 max_seq_length=max_seq_length,
                                                                 doc_stride=128,
                                                                 max_query_length=64,
                                                                 is_training=False,
                                                                 do_lower_case=False,
                                                                 debug=False)
        logger.info("Saving eval features into cached file %s", eval_feature_file)
        torch.save(eval_features, eval_feature_file)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in eval_features], dtype=torch.float)
    all_mask_start = torch.tensor([f.mask_start for f in eval_features], dtype=torch.float)
    all_example_index = torch.tensor([f.example_index for f in eval_features], dtype=torch.long)
    all_unique_ids = torch.tensor([f.unique_id for f in eval_features], dtype=torch.long)
    eval_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                 all_p_mask, all_mask_start, all_example_index, all_unique_ids)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", eval_batch_size)
    all_results = []

    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=True):
        if len(all_results) % 100 == 0:
            logger.info("evaluating %d/%d examples" % (len(all_results), len(eval_dataset)))
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2]
                      }

            inputs['p_mask'] = batch[3]
            unique_ids = batch[6]
            outputs = model(**inputs)

        for i, unique_id in enumerate(unique_ids):
            result = RawResultV2J(
                unique_id=int(unique_id),
                start_top_log_probs=to_list(outputs[0][i].view(-1)),
                start_top_index=to_list(outputs[1][i].view(-1)),
                end_top_log_probs=to_list(outputs[2][i].view(-1)),
                end_top_index=to_list(outputs[3][i].view(-1)),
                cls_logits=to_list(outputs[4][i]))
            all_results.append(result)

    result_dict = {}
    squad_utils.accumulate_predictions_v2j(
        result_dict, squad_examples, eval_features,
        all_results, 64, 5, 5)

    return squad_utils.write_predictions_v2j(
        result_dict, squad_examples, eval_features,
        all_results, n_best_size=20, max_answer_length=None,
        null_score_diff_threshold=None)