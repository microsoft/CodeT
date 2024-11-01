import ast
import ipdb

from ast_visitors import APIDefineVisitor, APICallVisitor, APIImportVisitor
from config import REPO_PACKAGE_DIR


class FileCallAPI:
    def __init__(self, repo, source_code_files):
        self.repo = repo
        self.source_code_files = source_code_files
        self.api_calls_by_file = dict()
    
    def _ast_processor_call(self, code, fpath_tuple):
        visitor = APICallVisitor(fpath_tuple)
        visitor.visit(ast.parse(code))
        return visitor
    
    def get_called_apis_by_file(self):
        print(f'Collecting called APIs in {self.repo}')
        for fpath_tuple, code in self.source_code_files.items():
            try:
                visitor = self._ast_processor_call(code, fpath_tuple)
            except Exception as e:
                print(f'{fpath_tuple} fail to parse: {e}')
                continue
            self.api_calls_by_file[fpath_tuple] = visitor.called_apis
        return self.api_calls_by_file


class FileDefinedAPI:
    def __init__(self, repo, source_code_files):
        self.repo = repo
        self.source_code_files = source_code_files
        self.defined_apis_by_file = dict()

    def _ast_processor_define(self, code, fpath_tuple):
        tree = ast.parse(code)
        visitor = APIDefineVisitor(fpath_tuple)
        visitor.store_parent_node(tree)
        visitor.visit(tree)
        return visitor

    def get_defined_apis_by_file(self):
        '''
        find defined apis in each python file
        '''
        print(f"Finding defined APIs in {self.repo}")
        for fpath_tuple, code in self.source_code_files.items():
            try:
                visitor = self._ast_processor_define(code, fpath_tuple)
            except Exception as e:
                print(f'{fpath_tuple} fail to parse: {e}')
                continue
            self.defined_apis_by_file[fpath_tuple] = {
                'defined_classes': visitor.defined_classes,  # dict
                'defined_outer_apis': visitor.defined_outer_apis,  # list
            }
        return self.defined_apis_by_file

class FileImportedAPI:
    def __init__(self, repo, source_code_files, defined_apis_by_file):
        self.repo = repo
        self.source_code_files = source_code_files
        self.defined_apis_by_file = defined_apis_by_file
        self.imported_apis_by_file = dict()
        
    def _ast_processor_import(self, code, file_tuple):
        file_module = build_file_module_from_file_tuple(self.repo, file_tuple)
        tree = ast.parse(code)
        visitor = APIImportVisitor(file_module, file_tuple)
        visitor.visit(tree)
        return visitor

    def get_imported_apis_by_file(self):
        '''
        find imported apis in each python file
        '''
        print(f"Finding imported APIs in {self.repo}")
        for fpath_tuple, code in self.source_code_files.items():
            try:
                visitor = self._ast_processor_import(code, fpath_tuple)
            except Exception as e:
                print(f'{fpath_tuple} fail to parse: {e}')
                continue
            # tring to locate the module of the imported api and the type of api (class or outer func)
            self.imported_apis_by_file[fpath_tuple] = self._get_apis_info(visitor.imported_apis)
        return self.imported_apis_by_file
    
    def _get_apis_info(self, imported_apis):
        '''
        tring to locate the module of the imported api and the type of api (class or outer func)
        '''
        imported_classes = list()
        imported_outer_apis = list()
        # those that are hard to decide the type (an init file refers to an init that refers to an init...)
        imported_members = list()
        imported_modules = list()
        for imported_api in imported_apis:
            imported_api_info = self._map_imported_api_to_fpath_tuple(imported_api)
            if not imported_api_info:  # not an intra-project api
                continue
            imported_api_type = imported_api_info['type']
            located_module_path_tuple = imported_api_info['module_path_tuple']
            module_path = imported_api_info['module_path']  # foo.bar
            if imported_api_type == 'module':
                imported_modules.append({
                    'module_name': module_path,
                    'located_module_path_tuple': located_module_path_tuple
                })
            elif imported_api_type == 'member':
                located_module_defined_apis = self.defined_apis_by_file[located_module_path_tuple]
                api_name = imported_api_info['api_name']
                if api_name in located_module_defined_apis['defined_classes']:
                    imported_classes.append({
                        'class_name': api_name,
                        'located_module_path_tuple': located_module_path_tuple
                    })
                elif api_name in located_module_defined_apis['defined_outer_apis']:
                    imported_outer_apis.append({
                        'api_name': api_name,
                        'located_module_path_tuple': located_module_path_tuple
                    })
                else:  # TODO: can not handle recursive import now
                    imported_members.append({
                        'member_name': api_name,
                        'located_module_path_tuple': located_module_path_tuple
                    })
        return {
            'imported_classes': imported_classes,
            'imported_outer_apis': imported_outer_apis,
            'imported_modules': imported_modules,
            'imported_members': imported_members  # not necessarily a callable api
        }
        

    def _map_imported_api_to_fpath_tuple(self, imported_api):
        '''
        return the most possible file module for the imported api
        '''
        def __find_possible_fpath_tuple(imported_node, current_fpath_tuple):
            located_file_tuples = [
                fpath_tuple for fpath_tuple in self.defined_apis_by_file.keys()
                if f'.{build_file_module_from_file_tuple(self.repo, fpath_tuple)}'.endswith(f'.{imported_node}')
            ]
            if len(located_file_tuples) == 1:
                return located_file_tuples[0]
            elif len(located_file_tuples) < 1:
                return None
            elif len(located_file_tuples) > 1:
                # when multiple files are found, we need to find the most possible one
                score = [self._longest_common_subsequence('.'.join(current_fpath_tuple), '.'.join(fpath_tuple)) for fpath_tuple in located_file_tuples]
                max_score_index = score.index(max(score))
                if score.count(max(score)) > 1 and len(located_file_tuples) == 2 and len([i for i in located_file_tuples if i[-1] == '__init__.py']) != 0:
                    # choose the one without init when there are two files with the same score
                    return [i for i in located_file_tuples if i[-1] != '__init__.py'][0]
                if score.count(max(score)) > 1:
                    print(located_file_tuples)
                    print(imported_api)
                    ipdb.set_trace()
                return located_file_tuples[max_score_index]

        api_name = imported_api['api_name']  # imported api can be a member or a module
        api_path = imported_api['api_path']  # foo.bar
        current_fpath_tuple = imported_api['current_fpath_tuple']
        deepest_node = '.'.join([i for i in [api_path, api_name] if i])
        located_file_tuple = __find_possible_fpath_tuple(deepest_node, current_fpath_tuple)
        if located_file_tuple:
            # imported api is a module
            return {
                'type': 'module',
                'module_path_tuple': located_file_tuple,
                'module_path': deepest_node
            }
        if not api_path:  # imported api is not a intra-project module
            return None
        located_file_tuple = __find_possible_fpath_tuple(api_path, current_fpath_tuple)
        if not located_file_tuple:
            # imported api is not a intra-project module
            return None
        if api_name == '*':  # import the entire module
            return {
                'type': 'module',
                'module_path_tuple': located_file_tuple,
                'module_path': api_path
            }
        # imported api is a member and we can find the module
        return {
            'type': 'member',
            'module_path_tuple': located_file_tuple,
            'module_path': api_path,
            'api_name': api_name,
        }
    
    def _longest_common_subsequence(self, text1, text2):
        shorter, longer = text1, text2
        if len(text2) < len(text1):
            shorter, longer = text2, text1
        common_length = 0
        for i in range(len(shorter)):
            common_length += 1 if shorter[i] == longer[i] else 0
        return common_length


def build_file_module_from_file_tuple(repo, fpath_tuple):
    # fpath_tuple: (repo_name, 'webui', 'launch.py')
    assert fpath_tuple[0] == repo and fpath_tuple[-1].endswith('.py')
    fpath_tuple = list(fpath_tuple)
    module_name = fpath_tuple[-1][:-3]  # launch
    fpath_repo_excluded = fpath_tuple[1:]  # ('webui', 'launch.py')
    if REPO_PACKAGE_DIR[repo]:  # need to modify module path
        original_package_dirs = REPO_PACKAGE_DIR[repo][0]
        if not original_package_dirs[0] in fpath_repo_excluded:  # the package_dir is not in the file path
            module_list = fpath_repo_excluded[:-1] + [module_name]
        else:  # the package_dir is in the file path
            assert all([fpath_repo_excluded[i] == original_package_dirs[i] for i in range(len(original_package_dirs))])
            mapped_source_code_dir = REPO_PACKAGE_DIR[repo][1] + fpath_repo_excluded[len(original_package_dirs):]
            module_list = mapped_source_code_dir[:-1] + [module_name]  # ['launch'] if REPO_PACKAGE_DIR[repo][1] is ('webui', [])
    else:
        module_list = fpath_repo_excluded[:-1] + [module_name]  # ['webui', 'launch']
    
    if module_name == '__init__':  # (repo_name, 'webui', '__init__.py')
        module_list = module_list[:-1]  # ['webui']
    return '.'.join(module_list)  # webui.launch