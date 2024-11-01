import os
import glob
import ipdb
import pickle
import json
import tiktoken


class CodexTokenizer():
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("p50k_base")
    
    def tokenize(self, text):
        return self.tokenizer.encode_ordinary(text)

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)

class Tools:
    @staticmethod
    def read_code(fname):
        with open(fname, 'r', encoding='utf8') as f:
            return f.read()
    
    @staticmethod
    def load_pickle(fname):
        with open(fname, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def load_json(fname):
        with open(fname, 'r', encoding='utf8') as f:
            return json.load(f)
    
    @staticmethod
    def dump_pickle(obj, fname):
        with open(fname, 'wb') as f:
            pickle.dump(obj, f)
    
    @staticmethod
    def dump_jsonl(obj, fname):
        with open(fname, 'w', encoding='utf8') as f:
            for item in obj:
                f.write(json.dumps(item) + '\n')
    
    @staticmethod
    def dump_json(obj, fname):
        with open(fname, 'w', encoding='utf8') as f:
            json.dump(obj, f, indent=4)
    
    @staticmethod
    def load_jsonl(fname):
        with open(fname, 'r', encoding='utf8') as f:
            lines = []
            for line in f:
                lines.append(json.loads(line))
            return lines
    
    @staticmethod
    def trim_context(tokenizer, previous_context_lines, context_max_tokens):
        previous_total_lines = len(previous_context_lines)
        previous_context = '\n'.join(previous_context_lines)
        tokens = tokenizer.tokenize(previous_context)
        decoded_context_total_lines = tokenizer.decode(tokens).count('\n') + 1
        try:
            assert previous_total_lines == decoded_context_total_lines
        except AssertionError:
            ipdb.set_trace()
        trimmed_tokens = tokens[-context_max_tokens:]
        trimmed_context = tokenizer.decode(trimmed_tokens)
        trimed_context_total_lines = trimmed_context.count('\n') + 1
        trimed_context_start_lineno = previous_total_lines - trimed_context_total_lines  # 0-indexed lineno
        return trimmed_context.splitlines(), trimed_context_start_lineno

    @staticmethod
    def iterate_repository(base_dir, repo):
        pattern = os.path.join(f'{base_dir}/{repo}', "**", "*.py")
        files = glob.glob(pattern, recursive=True)

        skipped_files = []
        loaded_code_files = dict()
        base_dir_list = os.path.normpath(base_dir).split(os.sep)
        for fname in files:
            try:
                code = Tools.read_code(fname)
                fpath_tuple = tuple(os.path.normpath(fname).split(os.sep)[len(base_dir_list):])
                loaded_code_files[fpath_tuple]= code
            except Exception as e:
                skipped_files.append((fname, e))
                continue

        if len(skipped_files) > 0:
            print(f"Skipped {len(skipped_files)} out of {len(files)} files due to I/O errors")
            for fname, e in skipped_files:
                print(f"{fname}: {e}")
        return loaded_code_files

    @staticmethod
    def tokenize(code):
        tokenizer = CodexTokenizer()
        return tokenizer.tokenize(code)


