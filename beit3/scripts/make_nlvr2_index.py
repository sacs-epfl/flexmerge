import os
import sys

if len(sys.argv) != 3:
    print("Usage: python make_nlvr2_index.py <data_path> <sentencepiece_model_path>")
    sys.exit(1)

data_path = os.path.abspath(sys.argv[1])
sentencepiece_model_path = os.path.abspath(sys.argv[2])

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from beit3.datasets import NLVR2Dataset
from transformers import XLMRobertaTokenizer

tokenizer = XLMRobertaTokenizer(sentencepiece_model_path)

NLVR2Dataset.make_dataset_index(
    data_path=data_path, 
    tokenizer=tokenizer, 
    nlvr_repo_path=os.path.join(data_path, "nlvr"),
)