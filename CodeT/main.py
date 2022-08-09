# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import logging
import os

from src.postprocess import PostProcessor
from src.execution import evaluate_with_test_code, evaluate_with_test_cases
from src.io_utils import Tools
from src.agreement import DataManager, DualAgreement
from src.evaluation import pass_at_K, get_result_of_sorted_solutions

logging.basicConfig(
    format="SystemLog: [%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path_for_solution", type=str, help="model input file in .jsonl format")
    parser.add_argument("--predict_path_for_solution", type=str, help="model output file in .jsonl format")
    parser.add_argument("--source_path_for_test", type=str, help="model input file in .jsonl format")
    parser.add_argument("--predict_path_for_test", type=str, help="model output file in .jsonl format")
    parser.add_argument("--cache_dir", type=str, help="the directory to store the cache files")
    parser.add_argument("--timeout", type=float, default=0.1, help="how many seconds to wait during execution for each test case")
    parser.add_argument("--test_case_limit", type=int, default=5, help="first n test cases per sample")

    args = parser.parse_args()
    
    handled_solutions, task_count = PostProcessor.map_task_id_for_solution(args.predict_path_for_solution, args.source_path_for_solution)
    handled_test_cases = PostProcessor.map_task_id_for_test_case(args.predict_path_for_test, args.source_path_for_test)
    
    ground_truth_exec_result = evaluate_with_test_code(handled_solutions, timeout=args.timeout)
    dual_exec_result = evaluate_with_test_cases(handled_solutions, handled_test_cases, timeout=args.timeout, limit=args.test_case_limit)
    
    Tools.dump_pickle(os.path.join(args.cache_dir, 'ground_truth_exec_result.pkl'), ground_truth_exec_result)
    Tools.dump_pickle(os.path.join(args.cache_dir, 'dual_exec_result.pkl'), dual_exec_result)
    
    data_manager = DataManager(dual_exec_result, handled_solutions, handled_test_cases, args.test_case_limit)
    set_consistency = DualAgreement(data_manager)
    ranked_result = set_consistency.get_sorted_solutions_without_iter()
    logger.info('pass rates of ranked solutions')
    get_result_of_sorted_solutions(ground_truth_exec_result, ranked_result)
    logger.info('pass rates of random solutions')
    pass_at_K(ground_truth_exec_result)