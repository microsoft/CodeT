# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from concurrent.futures import as_completed, ProcessPoolExecutor
import numpy as np
import scipy
import tqdm
import os
import copy
import functools

from utils import Tools, FilePathBuilder, CONSTANTS

class SimilarityScore:
    @staticmethod
    def cosine_similarity(embedding_vec1, embedding_vec2):
        return 1 - scipy.spatial.distance.cosine(embedding_vec1, embedding_vec2)
    
    @staticmethod
    def jaccard_similarity(list1, list2):
        set1 = set(list1)
        set2 = set(list2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return float(intersection) / union

class CodeSearchWorker:
    def __init__(self, repo_embedding_lines, query_embedding_lines, output_path, sim_scorer, max_top_k, log_message):
        self.repo_embedding_lines = repo_embedding_lines  # list
        self.query_embedding_lines = query_embedding_lines  # list
        self.max_top_k = max_top_k
        self.sim_scorer = sim_scorer
        self.output_path = output_path
        self.log_message = log_message
    
    def _is_context_after_hole(self, repo_embedding_line, query_line):
        hole_fpath_tuple = tuple(query_line['metadata']['fpath_tuple'])
        context_is_not_after_hole = []
        for metadata in repo_embedding_line['metadata']:
            if tuple(metadata['fpath_tuple']) != hole_fpath_tuple:
                context_is_not_after_hole.append(True)
                continue
            # now we know that the repo line is in the same file as the hole
            if metadata['end_line_no'] <= query_line['metadata']['context_start_lineno']:
                context_is_not_after_hole.append(True)
                continue
            context_is_not_after_hole.append(False)
        return not any(context_is_not_after_hole)
        
    def _find_top_k_context(self, query_line):
        top_k_context = []
        query_embedding = np.array(query_line['data'][0]['embedding'])
        for repo_embedding_line in self.repo_embedding_lines:
            if self._is_context_after_hole(repo_embedding_line, query_line):
                continue
            repo_line_embedding = np.array(repo_embedding_line['data'][0]['embedding'])
            similarity_score = self.sim_scorer(query_embedding, repo_line_embedding)
            top_k_context.append((repo_embedding_line, similarity_score))
        top_k_context = sorted(top_k_context, key=lambda x: x[1], reverse=False)[-self.max_top_k:]
        return top_k_context

    def run(self):
        query_lines_with_retrieved_results = []
        for query_line in self.query_embedding_lines:
            new_line = copy.deepcopy(query_line)
            top_k_context = self._find_top_k_context(new_line)
            new_line['top_k_context'] = top_k_context
            query_lines_with_retrieved_results.append(new_line)
        Tools.dump_pickle(query_lines_with_retrieved_results, self.output_path)


class CodeSearchWrapper:
    def __init__(self, vectorizer, benchmark, repos, window_sizes, slice_sizes):
        self.vectorizer = vectorizer
        if vectorizer == 'one-gram':
            self.sim_scorer = SimilarityScore.jaccard_similarity
            self.vector_path_builder = FilePathBuilder.one_gram_vector_path
        elif vectorizer == 'ada002':
            self.sim_scorer = SimilarityScore.cosine_similarity
            self.vector_path_builder = FilePathBuilder.ada002_vector_path
        self.max_top_k = 20  # store 20 top k context for the prompt construction (top 10)
        self.repos = repos
        self.window_sizes = window_sizes
        self.slice_sizes = slice_sizes
        self.benchmark = benchmark
    
    def _run_parallel(self, query_window_path_builder, prediction_path_template=None):
        workers = []
        for window_size in self.window_sizes:
            for slice_size in self.slice_sizes:
                for repo in self.repos:
                    if prediction_path_template:
                        query_window_path = query_window_path_builder(
                            prediction_path_template.format(window_size=window_size, slice_size=slice_size),
                            repo, window_size
                        )
                    else:
                        query_window_path = query_window_path_builder(repo, window_size)
                    query_line_path = self.vector_path_builder(query_window_path)
                    repo_window_path = FilePathBuilder.repo_windows_path(repo, window_size, slice_size)
                    repo_embedding_path = self.vector_path_builder(repo_window_path)
                    output_path = FilePathBuilder.retrieval_results_path(query_line_path, repo_embedding_path, self.max_top_k)
                    repo_embedding_lines = Tools.load_pickle(repo_embedding_path)
                    query_embedding_lines = Tools.load_pickle(query_line_path)
                    log_message = f'repo: {repo}, window: {window_size}, slice: {slice_size}  {self.vectorizer}, max_top_k: {self.max_top_k}'
                    worker = CodeSearchWorker(repo_embedding_lines, query_embedding_lines, output_path, self.sim_scorer, self.max_top_k, log_message)
                    workers.append(worker)
        # process pool
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = {executor.submit(worker.run, ) for worker in workers}
            for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
                future.result()

    def search_baseline_and_ground(self):
        query_line_path_temp = functools.partial(FilePathBuilder.search_first_window_path, self.benchmark, CONSTANTS.rg)
        self._run_parallel(query_line_path_temp)
        query_line_path_temp = functools.partial(FilePathBuilder.search_first_window_path, self.benchmark, CONSTANTS.gt)
        self._run_parallel(query_line_path_temp)
    
    def search_prediction(self, mode, prediction_path_template):
        query_line_path_temp = functools.partial(FilePathBuilder.gen_first_window_path, self.benchmark, mode)
        self._run_parallel(query_line_path_temp, prediction_path_template)
