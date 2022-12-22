import torch
from transformers import DebertaConfig
from src.models.deberta_modeling import DebertaForQAJointly

model_dir = "deberta_xlarge_v1_j_2"
config_name = "deberta_xlarge_v1_j_2/config.json"
#config = DebertaConfig().from_json_file(config_name)
model = DebertaForQAJointly.from_pretrained(model_dir)
for name, para in model.named_parameters():
    print("%s"%name, end="\n")
