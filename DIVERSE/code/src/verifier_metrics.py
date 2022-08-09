import absl  # Here to have a nice missing dependency error message early on
import nltk  # Here to have a nice missing dependency error message early on
import numpy  # Here to have a nice missing dependency error message early on
import six  # Here to have a nice missing dependency error message early on
from rouge_score import rouge_scorer, scoring
import datasets
import pdb
import numpy as np
import scipy
from tqdm import tqdm

from utils import (
    GSM8KCase,
    GSM8KExample,
    TextEntailmentCase,
    TextEntailmentExample,
    convert_eval_sequences_to_cases,
    compute_results,
    compute_results_avg,
)


case_class_map = {
    "GSM8K": GSM8KCase,
    "CLUTRR": TextEntailmentCase,
    "strategyQA": TextEntailmentCase,
}

example_class_map = {
    "GSM8K": GSM8KExample,
    "CLUTRR": TextEntailmentExample,
    "strategyQA": TextEntailmentExample,
}

_CITATION = ""
_DESCRIPTION = ""
_KWARGS_DESCRIPTION = ""


def simple_accuracy(preds, labels):
    correct_case_num = 0
    for pred, label in zip(preds, labels):
        pred = pred.replace(" ", "")
        label = label.replace(" ", "")
        if pred == label:
            correct_case_num += 1
    return correct_case_num / len(preds)


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class VerifierMetrics(datasets.Metric):
    def __init__(self, eval_sequences=None, pred_num_per_case=None, dataset_name=None, **kwargs,):
        super().__init__(**kwargs)
        self.pred_num_per_case = pred_num_per_case
        self.cases = convert_eval_sequences_to_cases(
            eval_sequences=eval_sequences,
            pred_num_per_case=pred_num_per_case,
            case_class=case_class_map[dataset_name],
            example_class=example_class_map[dataset_name],
        )
    
    def assign_scores(self, predictions):
        for i in range(0, len(predictions), self.pred_num_per_case + 1):
            curr_case_index = i // (self.pred_num_per_case + 1)
            self.cases[curr_case_index].ground_truth.verifier_score = predictions[i]
            for j in range(0, self.pred_num_per_case):
                self.cases[curr_case_index].preds[j].verifier_score = predictions[i+j+1]

    def _compute(self, predictions=None, references=None):
        self.assign_scores(predictions)
        result = {}
        result.update(compute_results_avg(self.cases, rand_k=100, repeat_time=10))
        result.update(compute_results_avg(self.cases, rand_k=75, repeat_time=10))
        result.update(compute_results_avg(self.cases, rand_k=50, repeat_time=10))
        result.update(compute_results_avg(self.cases, rand_k=25, repeat_time=10))
        result.update(compute_results_avg(self.cases, rand_k=20, repeat_time=10))
        result.update(compute_results_avg(self.cases, rand_k=10, repeat_time=10))
        result.update(compute_results_avg(self.cases, rand_k=5, repeat_time=10))
        result.update(compute_results_avg(self.cases, rand_k=2, repeat_time=10))
        return result

    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("float32", id="scores"),
                    "references": datasets.Value("float32", id="scores"),
                }
            ),
            codebase_urls=[],
            reference_urls=[],
        )
    
    def _metric_info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string", id="sequence"),
                    "references": datasets.Value("string", id="sequence"),
                }
            ),
            codebase_urls=[],
            reference_urls=[],
        )