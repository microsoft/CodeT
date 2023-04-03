# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tqdm
import itertools
from collections import defaultdict
from concurrent.futures import as_completed, ProcessPoolExecutor

from utils import Tools, FilePathBuilder, CONSTANTS

class BagOfWords:
    def __init__(self, input_file):
        self.input_file = input_file

    def build(self):
        print(f'building one gram vector for {self.input_file}')
        futures = dict()
        lines = Tools.load_pickle(self.input_file)
        with ProcessPoolExecutor(max_workers=48) as executor:
            for line in lines:
                futures[executor.submit(Tools.tokenize, line['context'])] = line
        
            new_lines = []
            t = tqdm.tqdm(total=len(futures))
            for future in as_completed(futures):
                line = futures[future]
                tokenized = future.result()
                new_lines.append({
                    'context': line['context'],
                    'metadata': line['metadata'],
                    'data': [{'embedding': tokenized}]
                })
                tqdm.tqdm.update(t)
            output_file_path = FilePathBuilder.one_gram_vector_path(self.input_file)
            Tools.dump_pickle(new_lines, output_file_path)


class BuildVectorWrapper:
    def __init__(self, benchmark, vector_builder, repos, window_sizes, slice_sizes):
        self.repos = repos
        self.window_sizes = window_sizes
        self.slice_sizes = slice_sizes
        self.vector_builder = vector_builder
        self.benchmark = benchmark

    def vectorize_repo_windows(self):
        for window_size, slice_size in itertools.product(self.window_sizes, self.slice_sizes):
            for repo in self.repos:
                builder = self.vector_builder(
                    FilePathBuilder.repo_windows_path(repo, window_size, slice_size)
                )
                builder.build()

    def vectorize_baseline_and_ground_windows(self):
        for window_size in self.window_sizes:
            for repo in self.repos:
                builder = self.vector_builder(FilePathBuilder.search_first_window_path(self.benchmark, CONSTANTS.rg, repo, window_size))
                builder.build()
                builder = self.vector_builder(FilePathBuilder.search_first_window_path(self.benchmark, CONSTANTS.gt, repo, window_size))
                builder.build()

    def vectorize_prediction_windows(self, mode, prediction_path_template):
        for window_size, slice_size in itertools.product(self.window_sizes, self.slice_sizes):
            prediction_path = prediction_path_template.format(window_size=window_size, slice_size=slice_size)
            for repo in self.repos:
                window_path = FilePathBuilder.gen_first_window_path(
                    self.benchmark, mode, prediction_path, repo, window_size
                )
                builder = self.vector_builder(window_path)
                builder.build()

class BuildEmbeddingVector:
    '''
    utilize external embedding model to generate embedding vector
    '''
    def __init__(self, repos, window_sizes, slice_sizes):
        self.repos = repos
        self.window_sizes = window_sizes
        self.slice_sizes = slice_sizes

    def build_input_file_for_repo_window(self, slice_size):
        lines = []
        for window_size in self.window_sizes:
            for repo in self.repos:
                file_path = FilePathBuilder.repo_windows_path(repo, window_size, slice_size)
                loaded_lines = Tools.load_pickle(file_path)
                for line in loaded_lines:
                    lines.append({
                        'context': line['context'],
                        'metadata': {
                            'window_file_path': file_path,
                            'original_metadata': line['metadata'],
                        },})
        return lines

    def build_input_file_search_first_window(self, mode, benchmark):
        lines = []
        for window_size in self.window_sizes:
            for repo in self.repos:
                file_path = FilePathBuilder.search_first_window_path(benchmark, mode, repo, window_size)
                loaded_lines = Tools.load_pickle(file_path)
                for line in loaded_lines:
                    lines.append({
                        'context': line['context'],
                        'metadata': {
                            'window_file_path': file_path,
                            'original_metadata': line['metadata']
                        }})
        return lines
    
    def build_input_file_for_gen_first_window(self, mode, benchmark, prediction_path):
        lines = []
        for window_size in self.window_sizes:
            for repo in self.repos:
                file_path = FilePathBuilder.gen_first_window_path(benchmark, mode, prediction_path, repo, window_size)
                loaded_lines = Tools.load_pickle(file_path)
                for line in loaded_lines:
                    lines.append({
                        'context': line['context'],
                        'metadata': {
                            'window_file_path': file_path,
                            'original_metadata': line['metadata']
                        }})
        return lines

    @staticmethod
    def place_generated_embeddings(generated_embeddings):
        vector_file_path_to_lines = defaultdict(list)
        for line in generated_embeddings:
            window_path = line['metadata']['window_file_path']
            original_metadata = line['metadata']['original_metadata']
            vector_file_path = FilePathBuilder.ada002_vector_path(window_path)
            vector_file_path_to_lines[vector_file_path].append({
                'context': line['context'],
                'metadata': original_metadata,
                'data': line['data']
            })
        for vector_file_path, lines in vector_file_path_to_lines.items():
            Tools.dump_pickle(lines, vector_file_path)
