# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import itertools
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from make_window import MakeWindowWrapper
from build_vector import BuildVectorWrapper, BagOfWords
from search_code import CodeSearchWrapper
from build_prompt import BuildPromptWrapper

from utils import CONSTANTS, CodexTokenizer, CodeGenTokenizer

def make_repo_window(repos, window_sizes, slice_sizes):
    MakeWindowWrapper(None, repos, window_sizes, slice_sizes).window_for_repo_files()
    vectorizer = BagOfWords
    BuildVectorWrapper(None, vectorizer, repos, window_sizes, slice_sizes).vectorize_repo_windows()


def run_RG1_and_oracle_method(benchmark, repos, window_sizes, slice_sizes):
    # build code snippets for vanilla retrieval-augmented approach and ground truth
    MakeWindowWrapper(benchmark, repos, window_sizes, slice_sizes).window_for_baseline_and_ground()
    # build vector for vanilla retrieval-augmented approach and ground truth
    vectorizer = BagOfWords
    BuildVectorWrapper(benchmark, vectorizer, repos, window_sizes, slice_sizes).vectorize_baseline_and_ground_windows()
    # search code for vanilla retrieval-augmented approach and ground truth
    CodeSearchWrapper('one-gram', benchmark, repos, window_sizes, slice_sizes).search_baseline_and_ground()
    # build prompt for vanilla retrieval-augmented approach and ground truth
    tokenizer = CodeGenTokenizer
    
    for window_size, slice_size in itertools.product(window_sizes, slice_sizes):
        mode = CONSTANTS.rg
        output_file_path = f'prompts/rg-one-gram-ws-{window_size}-ss-{slice_size}.jsonl'
        BuildPromptWrapper('one-gram', benchmark, repos, window_size, slice_size, tokenizer).build_first_search_prompt(mode, output_file_path)

        mode = CONSTANTS.gt
        output_file_path = f'prompts/gt-one-gram-ws-{window_size}-ss-{slice_size}.jsonl'
        BuildPromptWrapper('one-gram', benchmark, repos, window_size, slice_size, tokenizer).build_first_search_prompt(mode, output_file_path)


def run_RepoCoder_method(benchmark, repos, window_sizes, slice_sizes, prediction_path):
    mode = CONSTANTS.rgrg
    MakeWindowWrapper(benchmark, repos, window_sizes, slice_sizes).window_for_prediction(mode, prediction_path)
    vectorizer = BagOfWords
    BuildVectorWrapper(benchmark, vectorizer, repos, window_sizes, slice_sizes).vectorize_prediction_windows(mode, prediction_path)
    CodeSearchWrapper('one-gram', benchmark, repos, window_sizes, slice_sizes).search_prediction(mode, prediction_path)
    tokenizer = CodeGenTokenizer
    for window_size, slice_size in itertools.product(window_sizes, slice_sizes):
        output_file_path = f'prompts/repocoder-one-gram-ws-{window_size}-ss-{slice_size}.jsonl'
        BuildPromptWrapper('one-gram', benchmark, repos, window_size, slice_size, tokenizer).build_prediction_prompt(mode, prediction_path, output_file_path)


if __name__ == '__main__':
    repos = [
        'huggingface_diffusers',
        'nerfstudio-project_nerfstudio',
        'awslabs_fortuna',
        'huggingface_evaluate',
        'google_vizier',
        'alibaba_FederatedScope',
        'pytorch_rl',
        'opendilab_ACE',
    ]
    window_sizes = [20]
    slice_sizes = [2]  # 20 / 2 = 10

    # build window for the repos
    make_repo_window(repos, window_sizes, slice_sizes)
    
    # build prompt for the RG1 and oracle methods, after building the prompts, you should run inferece and then evaluate the results
    run_RG1_and_oracle_method(CONSTANTS.api_benchmark, repos, window_sizes, slice_sizes)

    '''
    before building prompt for the RepoCoder method, you need to run inference on the prompts of RG1 method
    '''
    # prediction_path = 'predictions/rg-one-gram-ws-20-ss-2_samples.0.jsonl'
    # run_RepoCoder_method(CONSTANTS.api_benchmark, repos, window_sizes, slice_sizes, prediction_path)