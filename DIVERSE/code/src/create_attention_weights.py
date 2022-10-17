

import os
import pickle
import numpy as np
import pandas as pd
import torch
import re
import spacy
from datasets import load_dataset, Dataset
import logging
import sys
from dataclasses import dataclass, field
from importlib import import_module
from typing import Dict, List, Optional, Tuple
import pdb
import numpy as np
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
import scipy
import shutil
import pickle
import transformers
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import is_main_process
from utils_ner import Split, TokenClassificationDataset, TokenClassificationTask
from deberta_model import DebertaV2ForTokenClassification
import pdb
import argparse

import utils_io


logger = logging.getLogger(__name__)


def get_args():

    parser = argparse.ArgumentParser(description="Training model parameters")


    parser.add_argument(
            "--train_data", type=str, default="/home/lyf/projects/aml-babel-components/datasets/gsm8k_debug/train.txt", 
            help="help")
    parser.add_argument(
            "--test_data", type=str, default="/home/lyf/projects/aml-babel-components/datasets/gsm8k_debug/test.txt", 
            help="help")
    parser.add_argument(
            "--labels", type=str, default="/home/lyf/projects/aml-babel-components/gsm8k/src/labels.txt", 
            help="help")
    parser.add_argument(
            "--output_dir", type=str, default="/home/lyf/projects/aml-babel-components/gsm8k/src/debug", 
            help="help")
    parser.add_argument(
            "--task_type", type=str, default="NER", 
            help="Task type to fine tune in training (e.g. NER, POS, etc)")
    parser.add_argument(
            "--model_name_or_path", type=str, default="microsoft/deberta-v3-large", 
            help="Path to pretrained model or model identifier from huggingface.co/models")     
    parser.add_argument(
            "--model_type", type=str, default="microsoft/deberta-v3-large", 
            help="Model to be used")
    parser.add_argument(
            "--max_seq_length", type=int, default=512, 
            help="max_seq_length")
    parser.add_argument(
            "--batch_size", type=int, default=8, 
            help="batch_size")
    parser.add_argument(
            "--random_seed", type=int, default=42, 
            help="Choose random seed")
    params, _ = parser.parse_known_args()

    return params


def find_desired_words_to_attend_to(train_dataset):
    # attention weights labeling
    desired_att_all_sentences = {}
    for i in range(len(train_dataset)):
        print("Calculating desired attention weights, example number:",i)
        input_ids = train_dataset[i].input_ids
        label_ids = train_dataset[i].label_ids
        input_ids_key = input_ids[:input_ids.index(2) + 1]  # find the <eos> token id
        label_key = label_ids[:input_ids.index(2) + 1]  # find the <eos> token id

        if label_key.count(3) > 0:  # find the "STEP-INCORRECT" label
            
            desired_words_to_attend = [0] * len(input_ids_key)
            idx = label_key.index(3)
            for x in range(idx, len(desired_words_to_attend)):
                desired_words_to_attend[x] = 1
            # pdb.set_trace()
        elif label_key.count(4) == 0:
            desired_words_to_attend = [1] * len(input_ids_key)

        assert len(input_ids_key) == len(desired_words_to_attend)

        desired_att_all_sentences.update({str(input_ids_key): desired_words_to_attend})
    return desired_att_all_sentences


if __name__ == "__main__":
    module = import_module("tasks")

    params_training = get_args()

    print(params_training)

    data_dir = os.path.join(params_training.output_dir, "data/")
    print("[data_dir]:", data_dir)
    os.makedirs(data_dir, exist_ok=True)

    shutil.copy(utils_io.get_file(params_training.train_data), data_dir)
    print(f"train file copied to: {data_dir}")
    shutil.copy(utils_io.get_file(params_training.test_data), data_dir + "dev.txt")
    print(f"dev file copied to: {data_dir}")
    shutil.copy(utils_io.get_file(params_training.test_data), data_dir)
    print(f"test file copied to: {data_dir}")
    shutil.copy(utils_io.get_file(params_training.labels), data_dir)
    print(f"labels file copied to: {data_dir}")

    try:
        token_classification_task_clazz = getattr(module, params_training.task_type)
        token_classification_task: TokenClassificationTask = token_classification_task_clazz()
    except AttributeError:
        raise ValueError(
            f"Task {params_training.task_type} needs to be defined as a TokenClassificationTask subclass in {module}. "
            f"Available tasks classes are: {TokenClassificationTask.__subclasses__()}"
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Prepare CONLL-2003 task
    labels = token_classification_task.get_labels(params_training.labels)
    label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}
    num_labels = len(labels)

    config = AutoConfig.from_pretrained(
        params_training.model_name_or_path,
        num_labels=num_labels,
        id2label=label_map,
        label2id={label: i for i, label in enumerate(labels)},
    )

    tokenizer = AutoTokenizer.from_pretrained(
        params_training.model_name_or_path,
    )

    # Get datasets
    train_dataset = (
        TokenClassificationDataset(
            token_classification_task=token_classification_task,
            data_dir=data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=params_training.max_seq_length,
            mode=Split.train,
        )
    )

    weights = find_desired_words_to_attend_to(train_dataset)
    # saving to file
    with open(os.path.join(params_training.output_dir, "attention_weights_deberta_all_sentences_saved.txt"), "wb") as fp:
        pickle.dump(weights, fp)

        # # For explanations we do not want to use, the first value is -1
        # if sum(desired_words_to_attend) == 0 \
        #         and desired_words_to_attend[0] != -1:
        #     desired_words_to_attend[0] = -1
    


