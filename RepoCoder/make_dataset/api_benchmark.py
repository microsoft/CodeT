import os
import ipdb
import random
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import as_completed, ProcessPoolExecutor

from file_visitors import FileDefinedAPI, FileImportedAPI, FileCallAPI
from make_dataset_utils import Tools, CodexTokenizer


class APICallLocator:
    def __init__(self, base_dir, repo):
        self.base_dir = base_dir
        self.repo = repo
        self.source_code_files = Tools.iterate_repository(base_dir, repo)
    
    def collect_defined_apis_for_each_file(self):
        file_define_api = FileDefinedAPI(self.repo, self.source_code_files)
        defined_apis_by_file = file_define_api.get_defined_apis_by_file()
        
        init_files = dict()
        for fpath_tuple in self.source_code_files.keys():
            if fpath_tuple[-1] == '__init__.py':
                init_files[fpath_tuple] = self.source_code_files[fpath_tuple]
        file_import_api = FileImportedAPI(self.repo, init_files, defined_apis_by_file)
        imported_apis_of_init_files = file_import_api.get_imported_apis_by_file()
        for module_path_tuple, imported_apis_info in imported_apis_of_init_files.items():
            defined_apis_info = defined_apis_by_file[module_path_tuple]
            defined_apis_by_file[module_path_tuple] = {**defined_apis_info, **imported_apis_info}
        return defined_apis_by_file
    
    def collect_available_apis_for_each_file(self):
        available_apis_by_file = self.collect_defined_apis_for_each_file()
        non_init_files = dict()
        for fpath_tuple in self.source_code_files.keys():
            if fpath_tuple[-1] != '__init__.py':
                non_init_files[fpath_tuple] = self.source_code_files[fpath_tuple]
        file_import_api = FileImportedAPI(self.repo, non_init_files, available_apis_by_file)
        imported_apis_of_non_init_files = file_import_api.get_imported_apis_by_file()
        for module_path_tuple, imported_apis_info in imported_apis_of_non_init_files.items():
            defined_apis_info = available_apis_by_file[module_path_tuple]
            available_apis_by_file[module_path_tuple] = {**defined_apis_info, **imported_apis_info}
        return available_apis_by_file
    
    def _build_func_signature_context_with_positions(self, base_dir, fpath_tuple, func_header_start_line_no, func_body_start_line_no, class_name):
        file_path = os.path.join(base_dir, *fpath_tuple)
        code = Tools.read_code(file_path)
        func_signature_and_doc = code.splitlines()[func_header_start_line_no-1:func_body_start_line_no-1]  # lineno is 1-indexed
        intent = 0
        if not func_signature_and_doc:
            ipdb.set_trace()
        for i in func_signature_and_doc[0]:
            if i == ' ': intent += 1
            else: break
        func_signature_and_doc = [i[intent:] for i in func_signature_and_doc]
        if class_name:
            func_signature_and_doc = [f'class {class_name}:'] + func_signature_and_doc
        return '\n'.join(func_signature_and_doc)

    def _build_func_body_context_with_positions(self, base_dir, fpath_tuple, func_start_line_no, func_end_line_no, class_name):
        file_path = os.path.join(base_dir, *fpath_tuple)
        code = Tools.read_code(file_path)
        func_body = code.splitlines()[func_start_line_no-1:func_end_line_no]  # lineno is 1-indexed
        intent = 0
        if not func_body:
            ipdb.set_trace()
        for i in func_body[0]:
            if i == ' ': intent += 1
            else: break
        func_body = [i[intent:] for i in func_body]
        if class_name:
            func_body = [f'class {class_name}:'] + func_body
        return '\n'.join(func_body)

    def _build_api_set_for_available_api_dicts(self, available_apis_by_file):
        def __buil_context_for_available_api(available_api):
            try:
                func_header_start_line_no = available_api['func_node_start_end_positions']['start_lineno']
                func_end_line_no = available_api['func_node_start_end_positions']['end_lineno']
                func_body_start_line_no = available_api['func_body_start_end_positions']['start_lineno'] if available_api['func_body_start_end_positions'] else func_end_line_no
                fpath_tuple = available_api['current_fpath_tuple']
                class_name = available_api['class_name'] if 'class_name' in available_api else None
                func_signature_context = self._build_func_signature_context_with_positions(self.base_dir, fpath_tuple, func_header_start_line_no, func_body_start_line_no, class_name)
                func_body_context = self._build_func_body_context_with_positions(self.base_dir, fpath_tuple, func_header_start_line_no, func_end_line_no, class_name)
            except Exception as e:
                print(e)
                ipdb.set_trace()

            return (available_api['api_name'], func_signature_context, func_body_context)
        
        available_api_set_by_file = defaultdict(set)
        for fpath_tuple in available_apis_by_file.keys():
            # imported apis, imported classes, imported modules, imported members
            outer_apis = set()
            outer_apis |= set([__buil_context_for_available_api(i) for i in available_apis_by_file[fpath_tuple]['imported_outer_apis']])
            
            class_apis = set()
            for class_info in available_apis_by_file[fpath_tuple]['imported_classes']:
                class_name = class_info['class_name']
                located_module_path_tuple = class_info['located_module_path_tuple']
                class_apis |= set([
                    __buil_context_for_available_api(i) for i in
                    available_apis_by_file[located_module_path_tuple]['defined_classes'][class_name]
                ])
            
            # TODO: cannot find the original position of the imported members from __init__
            # members = set([i['member_name'] for i in available_apis_by_file[fpath_tuple]['imported_members']])
            available_api_set_by_file[fpath_tuple] = outer_apis | class_apis
        
        for fpath_tuple in available_apis_by_file.keys():
            module_apis = set()
            imported_modules = [i['located_module_path_tuple'] for i in available_apis_by_file[fpath_tuple]['imported_modules']]
            for imported_module_path_tuple in imported_modules:
                module_apis |= available_api_set_by_file[imported_module_path_tuple]
            available_api_set_by_file[fpath_tuple] |= module_apis
        
        return available_api_set_by_file
    
    def find_intra_api_calls_for_each_file(self):
        available_apis_by_file = self.collect_available_apis_for_each_file()
        available_api_set_by_file = self._build_api_set_for_available_api_dicts(available_apis_by_file)
        file_call_api = FileCallAPI(self.repo, self.source_code_files)
        called_apis_by_file = file_call_api.get_called_apis_by_file()
        for fpath_tuple, called_apis_info in called_apis_by_file.items():
            available_api_set = available_api_set_by_file[fpath_tuple]
            called_intra_apis = []
            for called_api in called_apis_info:
                for available_api in available_api_set:
                    if called_api['api_name'] == available_api[0]:
                        called_api['signature_context'] = available_api[1]
                        called_api['body_context'] = available_api[2]
                        called_intra_apis.append(called_api)
                        break
            called_apis_by_file[fpath_tuple] = called_intra_apis
        return called_apis_by_file


class APIHoleDigger:
    def __init__(self, repo_base_dir, cache_base_dir, repo, context_max_tokens=2000):
        self.repo_base_dir = repo_base_dir
        self.repo = repo
        self.chosen_apis_cache_path = os.path.join(cache_base_dir, f'{self.repo}-random-api-200.pkl')
        self.context_max_tokens = context_max_tokens
        self.tokenizer = CodexTokenizer()

    def _make_context_prompt_by_prepending(self, base_dir, fpath_tuple, called_line_no, additional_context, context_max_tokens):
        # line_no is 0-indexed
        code = Tools.read_code(os.path.join(base_dir, *fpath_tuple))
        previous_code_lines = code.splitlines()[:called_line_no]
        if not previous_code_lines:
            ipdb.set_trace()
        additional_lines = []
        if additional_context:
            additional_lines = ["'''Relevant Helpful functions:"] + additional_context.splitlines() + ["'''"]
        trimed_context, context_start_lineno = Tools.trim_context(self.tokenizer, previous_code_lines, context_max_tokens)
        context_lines = additional_lines + trimed_context
        return '\n'.join(context_lines), context_start_lineno

    def _get_api_call_ground_truth(self, base_dir, fpath_tuple, start_line_no, end_line_no):
        code = Tools.read_code(os.path.join(base_dir, *fpath_tuple))
        code_lines = code.splitlines()
        return '\n'.join(code_lines[start_line_no:end_line_no+1])

    def _dig_hole(self, called_api, context_type):
        fpath_tuple = called_api['current_fpath_tuple']
        called_line_no = called_api['api_call_node_start_end_positions']['start_lineno'] - 1
        if context_type == 'none':
            additional_context  = ''
        elif context_type == 'signature':
            additional_context = called_api['signature_context']
        elif context_type == 'body':
            additional_context = called_api['body_context']
        context_prompt, context_start_lineno = self._make_context_prompt_by_prepending(self.repo_base_dir, fpath_tuple, called_line_no, additional_context, self.context_max_tokens)
        called_end_line_no = called_api['api_call_node_start_end_positions']['end_lineno'] - 1
        ground_truth = self._get_api_call_ground_truth(self.repo_base_dir, fpath_tuple, called_line_no, called_end_line_no)
        return context_prompt, context_start_lineno, ground_truth, fpath_tuple, called_line_no
    
    def dig_holes(self, context_type):
        chosen_apis = Tools.load_pickle(self.chosen_apis_cache_path)
        prompts = []
        print(f'digging holes for {self.repo}...')
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_to_prompt = {executor.submit(self._dig_hole, api, context_type): index for index, api in enumerate(chosen_apis)}
            for future in tqdm(as_completed(future_to_prompt), total=len(future_to_prompt)):
                index = future_to_prompt[future]
                prompt = future.result()
                prompts.append((prompt, index))
        prompts = sorted(prompts, key=lambda x: x[1])
        return [i[0] for i in prompts]
    
    def random_chosen(self, api_call_locator, num=200):
        if os.path.exists(self.chosen_apis_cache_path):
            return
        called_apis_by_file = api_call_locator.find_intra_api_calls_for_each_file()
        all_called_apis = [i for apis in list(called_apis_by_file.values()) for i in apis]
        random.shuffle(all_called_apis)
        Tools.dump_pickle(all_called_apis[:num], self.chosen_apis_cache_path)


def build_random_API_benchmark():
    REPO_BASE_DIR = 'downloaded_repos'
    OUT_BASE_DIR = 'output'
    CACHE_BASE_DIR = 'cache'
    repos = [
        'huggingface_diffusers',
        'nerfstudio-project_nerfstudio',
        'awslabs_fortuna',
        'huggingface_evaluate',
        'google_vizier',
        'alibaba_FederatedScope',
        'pytorch_rl',
        'opendilab_ACE'
    ]
    holedigger_by_repo = dict()
    for repo in repos:
        locator = APICallLocator(REPO_BASE_DIR, repo)
        holedigger = APIHoleDigger(REPO_BASE_DIR, CACHE_BASE_DIR, repo, context_max_tokens=2000)
        holedigger.random_chosen(locator)
        holedigger_by_repo[repo] = holedigger
    
    for context_type in ['none']:
        prompts_by_repo = dict()
        for repo in repos:
            holedigger = holedigger_by_repo[repo]
            prompts = holedigger.dig_holes(context_type)
            prompts_by_repo[repo] = prompts
        json_lines = []
        for repo, prompts in prompts_by_repo.items():
            json_lines.extend([
                {
                    'prompt': prompt, 
                    'metadata': {
                        'task_id': f'{repo}/{index}',
                        'ground_truth': ground_truth,
                        'fpath_tuple': fpath_tuple,
                        'context_start_lineno': context_start_lineno,
                        'line_no': end_line_no
                    }
                }
                for index, (prompt, context_start_lineno, ground_truth, fpath_tuple, end_line_no) in enumerate(prompts)
            ])
        Tools.dump_jsonl(json_lines, os.path.join(OUT_BASE_DIR, f'random-api-completion.jsonl'))

if __name__ == '__main__':
    build_random_API_benchmark()