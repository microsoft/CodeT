import re
from tqdm import tqdm
from multiset import Multiset
from functools import lru_cache
import random
import json
import pdb
import torch
import torch.nn.functional as F
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)
import time


class BaseCase:
    def __init__(self, ground_truth, preds):
        self.question = ""
        self.ground_truth = ground_truth
        self.preds = preds
        self.correct_preds_num = 0.0


class GSM8KCase(BaseCase):
    def __init__(self, ground_truth, preds):
        super().__init__(ground_truth, preds)
        self.entailment_batch_size = 512

    def do_step_labeling(self, model=None, tokenizer=None):
        # 将ground_truth标记为true
        self.ground_truth.is_correct = True
        for step in self.ground_truth.steps:
            self.ground_truth.step_labels[step] = 1

        # 先预存正样本集合
        positive_preds = [self.ground_truth]
        for i, pred in enumerate(self.preds):
            if pred.get_final_answer() != BaseExample.inf and pred.get_final_answer() == self.ground_truth.get_final_answer():
                positive_preds.append(pred)

        # 再对所有样本的所有step打标签
        for i, pred in enumerate(self.preds):
            if pred.get_final_answer() != BaseExample.inf and pred.get_final_answer() == self.ground_truth.get_final_answer():
                pred.is_correct = True
                for step in pred.steps:
                    pred.step_labels[step] = 1
            else:
                for k, step in enumerate(pred.steps):
                    ans = GSM8KExample.match(
                        pred.steps[:k+1],
                        positive_preds,
                        model=model,
                        tokenizer=tokenizer,
                    )
                    pred.step_labels[step] = ans


class TextEntailmentCase(BaseCase):
    def __init__(self, ground_truth, preds, entailment_batch_size=512):
        super().__init__(ground_truth, preds)
        self.entailment_results = {}
        self.entailment_batch_size = entailment_batch_size

    def do_step_labeling(self, model=None, tokenizer=None):
        # 将ground_truth标记为true
        self.ground_truth.is_correct = True
        for step in self.ground_truth.steps:
            self.ground_truth.step_labels[step] = 1

        # 先预存正样本集合
        positive_preds = [self.ground_truth]
        for i, pred in enumerate(self.preds):
            if pred.get_final_answer() != BaseExample.inf and pred.get_final_answer() == self.ground_truth.get_final_answer():
                positive_preds.append(pred)

        # 将所有待NLI的文本预存起来
        self.collect_entailment_texts(positive_preds)

        # print("Number of entailment result keys:", len(self.entailment_results.keys()))

        # 预处理所有NLI结果
        self.preprocess_entailment(model=model, tokenizer=tokenizer)

        # 再对所有样本的所有step打标签
        for i, pred in enumerate(self.preds):
            if pred.get_final_answer() != BaseExample.inf and pred.get_final_answer() == self.ground_truth.get_final_answer():
                pred.is_correct = True
                for step in pred.steps:
                    pred.step_labels[step] = 1
            else:
                for k, step in enumerate(pred.steps):
                    ans = TextEntailmentExample.match(
                        pred.steps[:k+1],
                        positive_preds,
                        model=model,
                        tokenizer=tokenizer,
                        entailment_result_dict=self.entailment_results,
                    )
                    pred.step_labels[step] = ans
    
    def collect_entailment_texts(self, positive_preds):
        for i, pred in enumerate(self.preds):
            if pred.get_final_answer() != BaseExample.inf and pred.get_final_answer() == self.ground_truth.get_final_answer():
                pass
            else:
                for pp in positive_preds:
                    for k, step in enumerate(pred.steps):
                        if k >= len(pp.steps):
                            continue
                        pp_step = pp.steps[k].strip()
                        text1 = f"premise: {pp_step} hypothesis: {step}"
                        text2 = f"premise: {step} hypothesis: {pp_step}"
                        self.entailment_results[text1] = -1
                        self.entailment_results[text2] = -1
    
    def preprocess_entailment(self, model, tokenizer):
        text_all = list(self.entailment_results.keys())
        text_batch, results_batch = [], []
        for i in range(0, len(text_all), self.entailment_batch_size):
            text_batch = text_all[i : min(len(text_all), i + self.entailment_batch_size)]
            batch_results = entailment_batch(text_batch, model, tokenizer)
            for sc in batch_results:
                results_batch.append(sc)
        for text, result in zip(text_batch, results_batch):
            self.entailment_results[text] = 1 if result else 0


class BaseExample:
    inf = "-99999999"
    
    def __init__(self, content):
        self.content = content.strip()
        self.steps = self.get_steps()
        self.step_labels = {}
        self.sequence_labels = []
        self.is_correct= False

    # Only for GSM8K dataset use
    def init_equations(self):
        raise NotImplementedError

    def get_steps(self):
        return [x+"%%" if x != self.content.split("%%")[-1] else x for i, x in enumerate(self.content.split("%%"))]

    def get_final_answer(self):
        ans = ""
        if "####" in self.content:
            ans = self.content.split("####")[-1].strip().replace("%%", "").replace(" ", "")
        else:
            ans = BaseExample.inf
        return clean_ans(ans)

    def label_to_string(self):
        return "".join(str(self.labels[k]) for k in self.labels.keys())


class GSM8KExample(BaseExample):
    def __init__(self, content):
        super().__init__(content)
        self.equations = self.init_equations()
        self.verifier_score = 0.0

    # 按'<<xxx>>'的格式将公式提取出来
    def init_equations(self):
        return [x for x in re.findall("<<.+>>[0-9\.]+", self.content) if "=" in x]

    def get_step_answer(step):
        expression = re.findall("<<.+>>[0-9\.]+", step)
        if len(expression == 0):
            ans = BaseExample.inf
        else:
            ans = expression[-1].split(">>")[-1].strip()
        return clean_ans(ans)
    
    @staticmethod
    @lru_cache(maxsize=4096)
    def get_answer(s):
        ans = ""
        if "####" in s:
            ans = s.split("####")[-1].replace("%%", "").replace(" ", "").strip()
        else:
            expression = re.findall("<<.+>>[0-9\.]+", s)
            if len(expression) == 0:
                ans = GSM8KExample.inf
            else:
                ans = expression[-1].split(">>")[-1].strip()
        return clean_ans(ans)
    
    @staticmethod
    def match(steps, positive_examples, model=None, tokenizer=None):
        curr_set = Multiset([GSM8KExample.get_answer(x) for x in steps])
        for positive_example in positive_examples:
            golden_set = Multiset([GSM8KExample.get_answer(x) for x in positive_example.steps])
            if GSM8KExample.inf in curr_set:
                curr_set.remove(GSM8KExample.inf)
            if GSM8KExample.inf in golden_set:
                golden_set.remove(GSM8KExample.inf)
            if len(curr_set) == 0:
                return 0
            if curr_set.issubset(golden_set):
                return 1
        return 0
    
    def get_sequence_labels(question, pred):
        sequence_labels = []
        if pred.is_correct:
            sequence_labels.append(("[CLS]", "SOLUTION-CORRECT"))
        else:
            sequence_labels.append(("[CLS]", "SOLUTION-INCORRECT"))

        # add step tokens
        for s in pred.steps:
            token_list = [x for x in re.split("(>>| )", s) if x != ' ']
            for token in token_list:
                if token == ">>":
                    if pred.step_labels[s] == 1:
                        sequence_labels.append((token, "STEP-CORRECT"))
                    else:
                        sequence_labels.append((token, "STEP-INCORRECT"))
                else:
                    sequence_labels.append((token, "O"))

        # add a split symbol
        sequence_labels.append(("&&", "O"))

        # add question tokens
        for token in question.split(" "):
            sequence_labels.append((token, "O"))

        return sequence_labels
    

class TextEntailmentExample(BaseExample):
    def __init__(self, content):
        super().__init__(content)

    @staticmethod
    def match(steps, positive_examples, model, tokenizer, entailment_result_dict):
        for pp in positive_examples:
            if TextEntailmentExample.match_per_example(pp, steps, entailment_result_dict):
                return 1
        return 0
    
    @staticmethod
    def match_per_example(pp, steps, entailment_result_dict):
        for k, step in enumerate(steps):
            if k >= len(pp.steps):
                continue
            # print("step:", step)
            # print("pp.steps[k]:", pp.steps[k])
            pp_step = pp.steps[k].strip()
            text1 = f"premise: {step} hypothesis: {pp_step}"
            text2 = f"premise: {pp_step} hypothesis: {step}"
            if entailment_result_dict[text1] == 0 or entailment_result_dict[text2] == 0:
                # error_case = 'No, Christmas trees are not dissimilar to deciduous trees.%%Both Christmas trees and deciduous trees are types of trees.%%Both Christmas trees and deciduous trees have leaves.%%So the answer is no.#### no'
                # if error_case in text1 or error_case in text2:
                #     print("text1:", text1)
                #     print("text2:", text2)
                #     pdb.set_trace()
                return 0
        return 1

    def get_sequence_labels(question, pred):
        sequence_labels = []
        if pred.is_correct:
            sequence_labels.append(("[CLS]", "SOLUTION-CORRECT"))
        else:
            sequence_labels.append(("[CLS]", "SOLUTION-INCORRECT"))

        # add step tokens
        for s in pred.steps:
            token_list = [x for x in re.split("(%%| )", s) if x != ' ']
            for token in token_list:
                if token == "":
                    continue
                if token == "%%":
                    if pred.step_labels[s] == 1:
                        sequence_labels.append((token, "STEP-CORRECT"))
                    else:
                        sequence_labels.append((token, "STEP-INCORRECT"))
                else:
                    sequence_labels.append((token, "O"))

        # add a split symbol
        sequence_labels.append(("&&", "O"))

        # add question tokens
        for token in question.split(" "):
            sequence_labels.append((token, "O"))

        return sequence_labels


@torch.no_grad()
def entailment_batch(text, model, tokenizer):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt").to("cuda")
    labels = torch.tensor([1] * len(text)).to("cuda")
    outputs = model(**inputs, labels=labels)
    logits = outputs.logits
    ans_list = torch.argmax(F.softmax(logits, dim=-1), dim=-1).tolist()
    ans_list = [x == model.config.label2id["ENTAILMENT"] for x in ans_list]
    return ans_list


@torch.no_grad()
def entailment(premise, hypothesis, model, tokenizer):
    text = f"premise: {premise} hypothesis: {hypothesis}"
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(model.device)
    labels = torch.tensor([1]).to(model.device)
    outputs = model(**inputs, labels=labels)
    logits = outputs.logits
    ans = torch.argmax(F.softmax(logits, dim=-1)).item() == model.config.label2id["ENTAILMENT"]
    return ans


def convert_eval_sequences_to_cases(eval_sequences, pred_num_per_case, case_class, example_class):
    cases = []
    for i in range(0, len(eval_sequences), pred_num_per_case + 1):
        case = case_class("", [])
        # question, grount_truth = eval_sequences[i].split("&&")[0], eval_sequences[i].split("&&")[1]
        question, grount_truth = eval_sequences[i].split("&&")[1], eval_sequences[i].split("&&")[0]
        case.ground_truth = example_class(grount_truth)
        case.question = question
        for j in range(i+1, i+pred_num_per_case+1):
            # case.preds.append(GSM8KExample(eval_sequences[j].split("&&")[1]))
            case.preds.append(example_class(eval_sequences[j].split("&&")[0]))
        cases.append(case)
    # if example_class.__name__ == "TextEntailmentExample":
    #     cases = post_process_answer_clutrr(cases)
    return cases


def post_process_answer_clutrr_mapping(cases):
    print("before loading pipeline")
    classifier = pipeline("zero-shot-classification", device=0)
    print("after loading pipeline")
    print("post processing")
    candidate_labels = ['sister', 'son', 'aunt', 'granddaughter', 'father', 'grandfather', 'grandmother', 'mother-in-law', 'uncle', 'niece', 'mother', 'brother', 'daughter', 'nephew', 'grandson', 'son-in-law', 'father-in-law', 'daughter-in-law']
    for case_idx, case in tqdm(enumerate(cases)):
        gt_ans = case.ground_truth.get_final_answer()
        # skip StrategyQA task
        if gt_ans == "yes" or gt_ans == "no":
            break
        for pred in case.preds:
            pred_ans = pred.get_final_answer()
            if pred_ans != BaseExample.inf and pred_ans != gt_ans:
                outputs = classifier(pred_ans, candidate_labels)
                logits = outputs["scores"]
                labels = outputs["labels"]
                candidate_index = np.argmax(logits)
                most_similar_answer = labels[candidate_index]
                body = pred.content.split("####")[0]
                pred.content = body + "####" + most_similar_answer
                # pdb.set_trace()
    return cases
            

def post_process_answer_clutrr_cutoff(cases):
    candidate_labels = ['sister', 'son', 'aunt', 'granddaughter', 'father', 'grandfather', 'grandmother', 'mother-in-law', 'uncle', 'niece', 'mother', 'brother', 'daughter', 'nephew', 'grandson', 'son-in-law', 'father-in-law', 'daughter-in-law']
    for case_idx, case in tqdm(enumerate(cases)):
        gt_ans = case.ground_truth.get_final_answer()
        # skip StrategyQA task
        if gt_ans == "yes" or gt_ans == "no":
            break
        for pred in case.preds:
            pred_ans = pred.get_final_answer()
            if pred_ans not in candidate_labels:
                body = pred.content.split("####")[0]
                pred.content = body + "####" + BaseExample.inf
    return cases


def random_1_hit(gt_ans, preds):
    idx = random.randint(0, len(preds)-1)
    # random 1 acc
    pred0_ans = preds[idx].get_final_answer()
    return 1 if pred0_ans == gt_ans else 0


def recall_hit(gt_ans, preds):
    for pred in preds:
        if pred.get_final_answer() == gt_ans:
            return 1
    return 0


def voting_hit(gt_ans, preds):
    # voting acc
    answers = {}
    for pred in preds:
        if pred.get_final_answer() not in answers:
            answers[pred.get_final_answer()] = 0
        answers[pred.get_final_answer()] += 1
    answers = sorted(answers.items(), key=lambda x : x[1], reverse=True)
    for i in range(len(answers)):
        ans, ans_cnt = answers[i][0], answers[i][1]
        if ans != GSM8KExample.inf:
            return 1 if ans == gt_ans else 0
    return 0


def weighted_voting_hit(gt_ans, preds):
    # voting acc
    answers = {}
    for pred in preds:
        if pred.get_final_answer() not in answers:
            answers[pred.get_final_answer()] = 0
        answers[pred.get_final_answer()] += pred.verifier_score
    answers = sorted(answers.items(), key=lambda x : x[1], reverse=True)
    for i in range(len(answers)):
        ans, ans_cnt = answers[i][0], answers[i][1]
        if ans != GSM8KExample.inf:
            return 1 if ans == gt_ans else 0
    return 0


def verification_hit(gt_ans, preds):
    preds = sorted(preds, key=lambda x : x.verifier_score, reverse=True)
    for pred in preds:
        ans = pred.get_final_answer()
        if ans != GSM8KExample.inf:
            return 1 if ans == gt_ans else 0
    return 0


def compute_top1_and_recall(data, rand_k=100):
    total_random_hit_cnt = 0
    total_vote_cnt = 0
    total_recall_cnt = 0
    for i, x in enumerate(data):
        gt_ans = x.ground_truth.get_final_answer()
        slice = x.preds if rand_k >= len(x.preds) else random.sample(x.preds, rand_k)
        
        total_random_hit_cnt += random_1_hit(gt_ans, slice)
        total_vote_cnt += voting_hit(gt_ans, slice)
        total_recall_cnt += recall_hit(gt_ans, slice)
    result = {
        "random_top1": total_random_hit_cnt / len(data), 
        "voting_top1_accuracy": total_vote_cnt / len(data),
        "recall": total_recall_cnt / len(data),
    }
    return result


def compute_results(data, rand_k=100):
    total_random_hit_cnt = 0
    total_recall_cnt = 0
    total_vote_cnt = 0
    total_weighted_vote_cnt = 0
    total_verification_cnt = 0
    for i, x in enumerate(data):
        gt_ans = x.ground_truth.get_final_answer()
        slice = x.preds if rand_k == len(x.preds) else random.sample(x.preds, rand_k)
        
        total_random_hit_cnt += random_1_hit(gt_ans, slice)
        total_vote_cnt += voting_hit(gt_ans, slice)
        total_recall_cnt += recall_hit(gt_ans, slice)
        total_weighted_vote_cnt += weighted_voting_hit(gt_ans, slice)
        total_verification_cnt += verification_hit(gt_ans, slice)
    result = {
        "random_top1": total_random_hit_cnt / len(data), 
        f"recall@{rand_k}": total_recall_cnt / len(data),
        f"verifier_top1_accuracy@{rand_k}": total_verification_cnt / len(data),
        f"voting_top1_accuracy@{rand_k}": total_vote_cnt / len(data),
        f"weighted_voting_top1_accuracy@{rand_k}": total_weighted_vote_cnt / len(data),
    }
    return result


def compute_results_avg(data, rand_k=100, repeat_time=5):
    sum_result_dict = {
        "random_top1": 0, 
        f"recall@{rand_k}": 0,
        f"verifier_top1_accuracy@{rand_k}": 0,
        f"voting_top1_accuracy@{rand_k}": 0,
        f"weighted_voting_top1_accuracy@{rand_k}": 0,
    }
    for i in tqdm(range(repeat_time)):
        for k in sum_result_dict:
            result_dict = compute_results(data, rand_k=rand_k)
            sum_result_dict[k] += result_dict[k]
    for k in sum_result_dict:
        sum_result_dict[k] = sum_result_dict[k] / repeat_time if repeat_time != 1 else sum_result_dict[k]
        sum_result_dict[k] = round(sum_result_dict[k], 8)
    return sum_result_dict
    

def dedup(li):
    s = set()
    new_li = []
    for x in li:
        if str(x) not in s:
            new_li.append(x)
            s.add(str(x))
    return new_li


def print_stat(data):
    cnt = 0
    for x in data:
        if x["output"] == "correct":
            cnt += 1
    print(cnt, len(data) - cnt, len(data))


def clean_ans(s):
    s = str(s)
    if s and len(s) > 0 and s[-1] == '.':
        s = s[:-1]
    return s.lower()  # for CLUTRR and strategyQA use