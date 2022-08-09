# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import statistics
import numpy as np
from collections import defaultdict
import logging
from typing import List, Union
import itertools

logging.basicConfig(
    format="SystemLog: [%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

def _dictionized_ground_truth_results(ground_truth_exec_results):
    ground_truth_results_by_task_and_solution = defaultdict(defaultdict)
    for result in ground_truth_exec_results:
        ground_truth_results_by_task_and_solution[result['task_id']][result['completion']] = result['passed']
    return ground_truth_results_by_task_and_solution

def _turn_solution_scores_into_choose_count(sorted_solution_scores, topk):
    # sorted_solution_scores: list of (solution, score)
    # if wrapped, sorted_solution_scores is list of ([solutions], score)
    # return list of (solution, choose_count)
    wrapped = True if type(sorted_solution_scores[0][0]) == list else False
    result = []
    if wrapped:
        last_score = sorted_solution_scores[0][1]
        merged_solutions_and_score = [sorted_solution_scores[0]]
        for solutions, score in sorted_solution_scores[1:]:
            if score == last_score:
                last_solutions = merged_solutions_and_score[-1][0]
                merged_solutions_and_score[-1] = (last_solutions + solutions, score)
            else:
                merged_solutions_and_score.append((solutions, score))
                last_score = score
        for solutions_and_score in merged_solutions_and_score:
            result.append((solutions_and_score[0], 1))  # choose one from solutions_and_score
    else:
        topk_scores = sorted(list(set([i[1] for i in sorted_solution_scores])), reverse=True)
        for score in topk_scores:
            solutions = [s[0] for s in sorted_solution_scores if s[1] == score]
            result.append((solutions, 1))

    if len(result) >= topk:
        return result[:topk]
    else:
        intial_choose_count = [1]*len(result)
        for i in range(topk-len(result)):
            intial_choose_count[i%len(result)] += 1
        for i, choose_count in enumerate(intial_choose_count):
            result[i] = (result[i][0], choose_count)
        return result
    

def get_result_of_sorted_solutions(ground_truth_results_list, sorted_solutions_by_task, topks=[1,2,10]):
    # sorted_solutions_by_task {task_id: [([solutions], score), ...]}
    def _count_correct(solutions: list, ground_truth_results: dict) -> int:
        return sum([ground_truth_results[s] for s in solutions])
    
    ground_truth_results = _dictionized_ground_truth_results(ground_truth_results_list)
    topk_results = dict()
    for topk in topks:
        random_pass_at_k_by_task = pass_at_K_by_task(ground_truth_results_list, k=topk)
        pass_rates = []
        for task_id in ground_truth_results.keys():
            all_wrong_probability = 1
            if task_id in sorted_solutions_by_task and sorted_solutions_by_task[task_id]:
                solutions_and_probability = _turn_solution_scores_into_choose_count(sorted_solutions_by_task[task_id], topk)
                for solutions, choose_count in solutions_and_probability:
                    current_wrong_prob = _estimator(len(solutions), _count_correct(solutions, ground_truth_results[task_id]), 1)
                    repeat_current_wrong_prob = pow(current_wrong_prob, choose_count)
                    all_wrong_probability *= repeat_current_wrong_prob
                pass_rates.append(1-all_wrong_probability)
            else:
                pass_rates.append(random_pass_at_k_by_task[task_id])
        
        # the avg rate of all tasks
        topk_results[f'pass@{topk}'] = round(statistics.mean(pass_rates), 4)
    logger.info(topk_results)

def pass_at_K_by_task(results, k):
    result_dict = defaultdict(list)
    for line in results:
        result_dict[line['task_id']].append(line['passed'])
    result = dict()
    for task_id in result_dict.keys():
        total = len(result_dict[task_id])
        correct = sum(result_dict[task_id])
        score = _estimate_pass_at_k(total, [correct], k)[0]
        result[task_id] = score
    return result

def pass_at_K(results, k = [1, 10, 100]):
    def _turn_list_into_dict(result_lines):
        result_dict = defaultdict(list)
        for line in result_lines:
            result_dict[line['task_id']].append(line['passed'])
        return result_dict

    # Calculate pass@k.
    total, correct = [], []
    for passed in _turn_list_into_dict(results).values():
        total.append(len(passed))
        correct.append(sum(passed))

    total = np.array(total)
    correct = np.array(correct)

    ks = k
    pass_at_k = {f"pass@{k}": round(_estimate_pass_at_k(total, correct, k).mean(), 4)
                 for k in ks if (total >= k).all()}
    logger.info(pass_at_k)

def _estimator(n: int, c: int, k: int) -> float:
    """
    Calculates comb(n - c, k) / comb(n, k).
    """
    if n - c < k:
        return 0
    return np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def _estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """
    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([1.0 - _estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])