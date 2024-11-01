import os
import random

from make_dataset_utils import Tools, CodexTokenizer


class RandomHoleDigger:
    def __init__(self, repo_base_dir, repo, context_max_tokens=2000, line_min_tokens=5, max_sample_per_repo=200):
        self.source_code_files = Tools.iterate_repository(repo_base_dir, repo)
        self.context_max_tokens = context_max_tokens
        self.max_sample_per_repo = max_sample_per_repo
        self.line_min_tokens = line_min_tokens
        self.repo = repo
        self.tokenizer = CodexTokenizer()
    
    def _get_line_types(self, lines):
        line_types = dict()
        in_multiline_comment = False
        multiline_comment_start = ""
        for lineno, line in enumerate(lines):
            stripped_line = line.strip()
            if not stripped_line:
                line_types[lineno] = 'empty'
                continue
            if in_multiline_comment:
                if stripped_line.endswith(multiline_comment_start):
                    in_multiline_comment = False
                line_types[lineno] = 'comment'
            elif stripped_line.startswith('"""') or stripped_line.startswith("'''"):
                in_multiline_comment = True
                multiline_comment_start = stripped_line[:3]
                line_types[lineno] = 'comment'
            elif stripped_line and stripped_line[0] == "#":
                line_types[lineno] = 'comment'
            else:
                line_types[lineno] = 'code'
        return line_types

    def _get_usable_lines(self, lines):
        line_types = self._get_line_types(lines)
        usable_lines = []
        for lineno, line_type in line_types.items():
            if line_type == 'code':
                if lineno == 0:
                    continue
                if line_types[lineno - 1] == 'empty':
                    continue
                usable_lines.append(lineno)
        return usable_lines

    def get_chosen_lines(self):
        candidate_lines = []
        for fpath_tuple, code in self.source_code_files.items():
            code_lines = code.splitlines()
            usable_lines = self._get_usable_lines(code_lines)
            candidate_lines.extend([(fpath_tuple, line_no) for line_no in usable_lines])
        random.shuffle(candidate_lines)
        chosen_lines = []
        chosen_line_strs = set()
        for fpath_tuple, line_no in candidate_lines:
            code_lines = self.source_code_files[fpath_tuple].splitlines()
            line = code_lines[line_no]
            if len(self.tokenizer.tokenize(line)) > self.context_max_tokens:
                continue
            if line.strip() in chosen_line_strs:
                continue
            chosen_line_strs.add(line.strip())
            chosen_lines.append({
                'fpath_tuple': fpath_tuple,
                'line_no': line_no,
                'ground_truth': line,
                'code_lines': code_lines
            })
            if len(chosen_lines) >= self.max_sample_per_repo:
                break
        return chosen_lines

    def _make_context(self, line):
        previous_lines = line['code_lines'][:line['line_no']]
        trimmed_lines, trimed_context_start_lineno = Tools.trim_context(self.tokenizer, previous_lines, self.context_max_tokens)
        return '\n'.join(trimmed_lines), trimed_context_start_lineno

    def make_dataset(self):
        test_data = []
        chosen_lines = self.get_chosen_lines()
        for index, line in enumerate(chosen_lines):
            prompt, trimed_context_start_lineno = self._make_context(line)
            test_data.append({
                'prompt': prompt,
                'metadata': {
                    'task_id':  f'{self.repo}/{index}',
                    'ground_truth': line['ground_truth'],
                    'fpath_tuple': line['fpath_tuple'],
                    'context_start_lineno': trimed_context_start_lineno,
                    'line_no': line['line_no']
            }})
        print(f'Generated {len(test_data)} samples for {self.repo}')
        return test_data

if __name__ == '__main__':
    OUT_BASE_DIR = 'output'
    REPO_BASE_DIR = 'downloaded_repos'
    repos = [
        'huggingface_diffusers',
        'nerfstudio-project_nerfstudio',
        'awslabs_fortuna',
        'huggingface_evaluate',
        'google_vizier',
        'PaddlePaddle_PaddleTS',
        'microsoft_RegionCLIP',
        'alibaba_FederatedScope',
        'pytorch_rl',
        'opendilab_ACE'
    ]
    lines = []
    for repo in repos:
        print(f'Processing {repo}')
        digger = RandomHoleDigger(REPO_BASE_DIR, repo)
        lines += digger.make_dataset()
    Tools.dump_jsonl(lines, os.path.join(OUT_BASE_DIR, 'ten-repos-random-line-completion.jsonl'))


