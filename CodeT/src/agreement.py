# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict, Counter
import logging
import math


logging.basicConfig(
    format="SystemLog: [%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self, dual_exec_results, sampled_code_by_task, sampled_test_case_by_task, limit):
        logger.info('handling dual exec results')
        self.dual_exec_results = dual_exec_results
        self.sampled_code_by_task = sampled_code_by_task
        self.sampled_test_case_by_task = sampled_test_case_by_task
        self.limit = limit
        
        self.solution_frequency_by_task = defaultdict(Counter)
        self.test_case_frequency_by_task = dict()
        self.passed_unique_solutions_by_task = defaultdict(set)
        self.passed_unique_test_cases_by_task = defaultdict(set)
        self.passed_solution_test_case_pairs_by_task = defaultdict(set)
        self.solution_string_to_id_range_by_task = dict()
        self.test_case_string_to_id_range_by_task = dict()
        self.solution_id_to_string_by_task = dict()
        self.test_case_id_to_string_by_task = dict()
        
        self.expanded_passed_solution_test_case_pairs_by_task = defaultdict(list)
        
        self._get_solution_frequency()
        logger.info('got solution frequency')
        self._get_test_case_frequency()
        logger.info('got test case frequency')
        self._get_passed_solution_test_case_pairs_by_task()
        logger.info('got passed solution test case pairs by task')
        self._get_solution_and_test_case_ids()
        logger.info('got solution and test case ids')
        self._get_expanded_dual_exec_result()
        logger.info('got expanded dual exec results')
        
    def _get_solution_frequency(self):
        for sample in self.sampled_code_by_task:
            task_id = sample['task_id']
            completion = sample['completion']
            self.solution_frequency_by_task[task_id][completion] += 1

    def _get_test_case_frequency(self):
        for task_id in self.sampled_test_case_by_task.keys():
            task_test_cases = [
                cases_per_sample[:self.limit] for cases_per_sample in self.sampled_test_case_by_task[task_id]
            ]
            task_test_cases = sum(task_test_cases, [])
            self.test_case_frequency_by_task[task_id] = Counter(task_test_cases)
    
    def _get_passed_solution_test_case_pairs_by_task(self):
        for result in self.dual_exec_results:
            if not result['passed']:
                continue
            for idx, test_case in enumerate(result['test_cases']):
                if result['result'][idx] != True:
                    continue
                if test_case not in self.test_case_frequency_by_task[result['task_id']]:
                    continue
                self.passed_solution_test_case_pairs_by_task[result['task_id']].add((result['completion'], test_case))
                self.passed_unique_solutions_by_task[result['task_id']].add(result['completion'])
                self.passed_unique_test_cases_by_task[result['task_id']].add(test_case)

    def _build_string_to_id_range(self, frequency_dict, limited_values):
        id_ranges = dict()
        start_id = 0
        for key, value in frequency_dict.items():
            if key not in limited_values:
                continue
            id_ranges[key] = range(start_id, start_id + value)
            start_id += value
        return id_ranges
    
    def _build_id_to_string(self, str_to_id_range):
        id_to_string = dict()
        for string in str_to_id_range.keys():
            for idx in str_to_id_range[string]:
                id_to_string[idx] = string
        return id_to_string
    
    def _get_solution_and_test_case_ids(self):
        for task_id in self.solution_frequency_by_task.keys():
            self.solution_string_to_id_range_by_task[task_id] = self._build_string_to_id_range(self.solution_frequency_by_task[task_id], self.passed_unique_solutions_by_task[task_id])
            self.test_case_string_to_id_range_by_task[task_id] = self._build_string_to_id_range(self.test_case_frequency_by_task[task_id], self.passed_unique_test_cases_by_task[task_id])
            self.solution_id_to_string_by_task[task_id] = self._build_id_to_string(self.solution_string_to_id_range_by_task[task_id])
            self.test_case_id_to_string_by_task[task_id] = self._build_id_to_string(self.test_case_string_to_id_range_by_task[task_id])
    
    def _get_expanded_by_id_range(self, solution_id_range, test_case_id_range):
        result = list()
        for solution_id in solution_id_range:
            for test_case_id in test_case_id_range:
                result.append((solution_id, test_case_id))
        return result
    
    def _get_expanded_dual_exec_result(self):
        for task_id in self.passed_solution_test_case_pairs_by_task.keys():
            for solution_str, test_case_str in self.passed_solution_test_case_pairs_by_task[task_id]:
                solution_id_range = self.solution_string_to_id_range_by_task[task_id][solution_str]
                test_case_id_range = self.test_case_string_to_id_range_by_task[task_id][test_case_str]
                self.expanded_passed_solution_test_case_pairs_by_task[task_id] += self._get_expanded_by_id_range(solution_id_range, test_case_id_range)


class DualAgreement:
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.dual_exec_results_by_task = data_manager.expanded_passed_solution_test_case_pairs_by_task
        self.solution_id_to_string_by_task = data_manager.solution_id_to_string_by_task
        
        self.solution_passed_cases_by_task = defaultdict(defaultdict)
        self.caseset_passed_solutions_by_task = defaultdict(defaultdict)
        
        self._get_solution_passed_case_set()
        logger.info('got solution passed case sets')
        self._get_caseset_passed_solutions()
        logger.info('got case set passed solutions')
    
    def _get_solution_passed_case_set(self):
        for task_id in self.dual_exec_results_by_task:
            for solution, test_case in self.dual_exec_results_by_task[task_id]:
                if solution in self.solution_passed_cases_by_task[task_id]:
                    self.solution_passed_cases_by_task[task_id][solution].append(test_case)
                else:
                    self.solution_passed_cases_by_task[task_id][solution] = [test_case]

    def _get_caseset_passed_solutions(self):
        for task_id in self.solution_passed_cases_by_task.keys():
            for solution in self.solution_passed_cases_by_task[task_id].keys():
                case_set = tuple(sorted(self.solution_passed_cases_by_task[task_id][solution]))  # case_set: set of (test_case, score)
                if case_set in self.caseset_passed_solutions_by_task[task_id]:
                    self.caseset_passed_solutions_by_task[task_id][case_set].append(solution)
                else:
                    self.caseset_passed_solutions_by_task[task_id][case_set] = [solution]
    
    def get_sorted_solutions_without_iter(self):
        logger.info('Start to get sorted solutions without iter')
        # caseset_passed_solutions = {task_id: {case_set: [solution]}}
        ranked_solutions_by_task = defaultdict(list)
        for task_id in self.caseset_passed_solutions_by_task.keys():
            flatted_case_set_passed_solutions = []
            for case_set in self.caseset_passed_solutions_by_task[task_id].keys():
                solution_set = self.caseset_passed_solutions_by_task[task_id][case_set]
                solution_set_score = math.sqrt(len(solution_set))
                case_set_score = len(case_set)
                solution_str_set = [self.solution_id_to_string_by_task[task_id][solution] for solution in solution_set]
                flatted_case_set_passed_solutions.append((solution_str_set, case_set_score*solution_set_score))
            ranked_solutions_by_task[task_id] = sorted(flatted_case_set_passed_solutions, key=lambda x: x[1], reverse=True)
        return ranked_solutions_by_task
