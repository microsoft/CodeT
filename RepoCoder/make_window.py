# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import itertools
import functools

from utils import Tools, FilePathBuilder, CONSTANTS
from collections import defaultdict

class RepoWindowMaker:
    def __init__(self, repo, window_size, slice_size):
        self.repo = repo
        self.window_size = window_size
        self.slice_size = slice_size
        self.slice_step = 1 if window_size // slice_size == 0 else window_size // slice_size
        self.source_code_files = Tools.iterate_repository(repo)
        
    def _buid_windows_for_a_file(self, fpath_tuple, code):
        code_windows = []
        code_lines = code.splitlines()
        delta_size = self.window_size // 2
        for line_no in range(0, len(code_lines), self.slice_step): # line_no starts from 0
            start_line_no = max(0, line_no - delta_size)
            end_line_no = min(len(code_lines), line_no + self.window_size - delta_size)
            window_lines = [i for i in code_lines[start_line_no:end_line_no]]
            if not window_lines:  # all empty lines
                continue
            window_text = '\n'.join(window_lines)
            code_windows.append({
                'context': window_text,
                'metadata': {
                    'fpath_tuple': fpath_tuple,
                    'line_no': line_no,
                    'start_line_no': start_line_no,
                    'end_line_no': end_line_no,
                    'window_size': self.window_size,
                    'repo': self.repo,
                    'slice_size': self.slice_size,
                }
            })
        return code_windows
    
    def _merge_windows_with_same_context(self, code_windows):
        merged_code_windows = defaultdict(list)
        for code_window in code_windows:
            context = code_window['context']
            metadata = code_window['metadata']
            merged_code_windows[context].append(metadata)
        json_lines = []
        for context, metadata_list in merged_code_windows.items():
            json_lines.append({
                'context': context,
                'metadata': metadata_list
            })
        return json_lines

    def build_windows(self):
        all_code_windows = []
        for fpath_tuple, code in self.source_code_files.items():
            all_code_windows += self._buid_windows_for_a_file(fpath_tuple, code)
        merged_code_windows = self._merge_windows_with_same_context(all_code_windows)
        print(f'build {len(merged_code_windows)} windows for {self.repo} with window size {self.window_size} and slice {self.slice_size}')
        output_path = FilePathBuilder.repo_windows_path(self.repo, self.window_size, self.slice_size)
        Tools.dump_pickle(merged_code_windows, output_path)


class BaselineWindowMaker:
    '''the retrieve-and-generate approach'''
    def __init__(self, benchmark, repo, window_size, tasks):
        self.benchmark = benchmark
        self.repo = repo
        self.window_size = window_size
        self.tasks = tasks
        self.source_code = Tools.iterate_repository(repo)
    
    def build_window(self):
        code_windows = []
        for task in self.tasks:
            if task['metadata']['task_id'].split('/')[0] != self.repo:
                continue
            fpath_tuple = tuple(task['metadata']['fpath_tuple'])
            line_no = task['metadata']['line_no']
            original_code = self.source_code[fpath_tuple]
            code_lines = original_code.splitlines()
            context_start_lineno = task['metadata']['context_start_lineno']
            start_line_no = max(context_start_lineno, line_no - self.window_size)
            window_lines = [i for i in code_lines[start_line_no:line_no]]
            code_windows.append({
                'context': '\n'.join(window_lines),
                'metadata': {
                    'fpath_tuple': fpath_tuple,
                    'line_no': line_no,  # line_no starts from 0
                    'task_id': task['metadata']['task_id'],
                    'start_line_no': start_line_no,
                    'end_line_no': line_no,
                    'window_size': self.window_size,
                    'context_start_lineno': context_start_lineno,
                    'repo': self.repo
                }
            })
        print(f'build {len(code_windows)} baseline windows for {self.repo} with window size {self.window_size}')
        output_path = FilePathBuilder.search_first_window_path(self.benchmark, CONSTANTS.rg, self.repo, self.window_size)
        Tools.dump_pickle(code_windows, output_path)

class GroundTruthWindowMaker:
    def __init__(self, benchmark, repo, window_size, tasks):
        self.benchmark = benchmark
        self.repo = repo
        self.window_size = window_size
        self.tasks = tasks
        self.source_code = Tools.iterate_repository(repo)

    def build_window(self):
        code_windows = []
        delta_size = self.window_size // 2
        for task in self.tasks:
            if task['metadata']['task_id'].split('/')[0] != self.repo:
                continue
            fpath_tuple = tuple(task['metadata']['fpath_tuple'])
            line_no = task['metadata']['line_no']
            original_code = self.source_code[fpath_tuple]
            code_lines = original_code.splitlines()
            context_start_lineno = task['metadata']['context_start_lineno']
            start_line_no = max(context_start_lineno, line_no - delta_size)
            end_line_no = min(len(code_lines), line_no + self.window_size - delta_size)
            window_lines = [i for i in code_lines[start_line_no:end_line_no]]
            code_windows.append({
                'context': '\n'.join(window_lines),
                'metadata': {
                    'fpath_tuple': fpath_tuple,
                    'line_no': line_no,  # line_no starts from 0
                    'task_id': task['metadata']['task_id'],
                    'start_line_no': start_line_no,
                    'end_line_no': end_line_no,
                    'window_size': self.window_size,
                    'context_start_lineno': context_start_lineno,
                    'repo': self.repo
                }
            })
        print(f'build {len(code_windows)} ground truth windows for {self.repo} with window size {self.window_size}')
        output_path = FilePathBuilder.search_first_window_path(self.benchmark, CONSTANTS.rg, self.repo, self.window_size)
        Tools.dump_pickle(code_windows, output_path)

class PredictionWindowMaker:
    def __init__(self, repo, window_size, prediction_path, window_path_builder):
        self.repo = repo
        self.window_size = window_size
        self.prediction_path = prediction_path
        self.source_code = Tools.iterate_repository(repo)
        self.predictions = Tools.load_jsonl(prediction_path)
        self.window_path_builder = window_path_builder
    
    def build_window(self, type='centered'):
        code_windows = []
        delta_size = self.window_size // 2
        for prediction in self.predictions:
            if prediction['metadata']['task_id'].split('/')[0] != self.repo:
                continue
            fpath_tuple = tuple(prediction['metadata']['fpath_tuple'])
            line_no = prediction['metadata']['line_no']  # line_no in prediction file starts from 0
            original_code = self.source_code[fpath_tuple]
            code_lines = original_code.splitlines()
            context_start_lineno = prediction['metadata']['context_start_lineno']
            start_line_no = max(context_start_lineno, line_no - delta_size)
            for sample in [prediction['choices'][i]['text'] for i in range(len(prediction['choices']))]:
                # TODO actually only one sample is generated
                sample_lines = [i for i in sample.splitlines() if i.strip()]
                new_code_lines = code_lines[:line_no] + sample_lines
                end_line_no = min(len(new_code_lines), line_no + self.window_size - delta_size)
                window_lines = [i for i in new_code_lines[start_line_no:end_line_no] if i.strip()]
                if not window_lines:  # all empty lines
                    continue
                code_windows.append({
                    'context': '\n'.join(window_lines),
                    'metadata': {
                        'fpath_tuple': fpath_tuple,
                        'line_no': line_no,  # line_no starts from 0
                        'prediction': sample,
                        'task_id': prediction['metadata']['task_id'],
                        'start_line_no': start_line_no,
                        'end_line_no': end_line_no,
                        'window_size': self.window_size,
                        'context_start_lineno': context_start_lineno,
                        'repo': self.repo
                    }
                })
        print(f'build {len(code_windows)} prediction windows for {self.repo} with window size {self.window_size}')
        output_path = self.window_path_builder(self.prediction_path, self.repo, self.window_size)
        Tools.dump_pickle(code_windows, output_path)

class MakeWindowWrapper:
    def __init__(self, benchmark, repos, window_sizes, slice_sizes):
        self.repos = repos
        self.window_sizes = window_sizes
        self.slice_sizes = slice_sizes

        self.benchmark = benchmark

        if benchmark == CONSTANTS.line_benchmark:
            self.task_file_path = FilePathBuilder.random_line_completion_benchmark
        elif benchmark == CONSTANTS.api_benchmark:
            self.task_file_path = FilePathBuilder.api_completion_benchmark
        elif benchmark == CONSTANTS.short_line_benchmark:
            self.task_file_path = FilePathBuilder.short_random_line_completion_benchmark
        elif benchmark == CONSTANTS.short_api_benchmark:
            self.task_file_path = FilePathBuilder.short_api_completion_benchmark

    def window_for_repo_files(self):
        for window_size, slice_size in itertools.product(self.window_sizes, self.slice_sizes):
            for repo in self.repos:
                repo_window_maker = RepoWindowMaker(repo, window_size, slice_size)
                repo_window_maker.build_windows()

    def window_for_baseline_and_ground(self):
        tasks = Tools.load_jsonl(self.task_file_path)
        for window_size in self.window_sizes:
            for repo in self.repos:
                baseline_window_maker = BaselineWindowMaker(self.benchmark, repo, window_size, tasks)
                ground_window_maker = GroundTruthWindowMaker(self.benchmark, repo, window_size, tasks)
                baseline_window_maker.build_window()
                ground_window_maker.build_window()

    def window_for_prediction(self, mode, prediction_path_template):
        for window_size, slice_size in itertools.product(self.window_sizes, self.slice_sizes):
            prediction_path = prediction_path_template.format(window_size=window_size, slice_size=slice_size)
            for repo in self.repos:
                window_path_builder = functools.partial(FilePathBuilder.gen_first_window_path, self.benchmark, mode)
                pred_window_maker = PredictionWindowMaker(repo, window_size, prediction_path, window_path_builder)
                pred_window_maker.build_window()
