"""
parse Python files to find:
1. the start and end locations of called apis
2. the definations of all apis and classes that can be called
"""
import ast
import astunparse
from collections import defaultdict
import ipdb

class APICallVisitor(ast.NodeVisitor):
    def __init__(self, fpath_tuple):
        super().__init__()
        # stores all the called apis, including intra-project and public ones
        self.called_apis = list()
        self.fpath_tuple = fpath_tuple
   
    def visit_Call(self, node: ast.AST):
        # start/ending positions for the api call(including api_prefix, api_name, and arguments)
        start_lineno = node.lineno
        start_col = node.col_offset
        end_lineno = node.end_lineno
        end_col = node.end_col_offset
        # lineno starts from 1, col_offset starts from 0, and end_offset contains the last symbol
        # to extract the segment, use line[start_col:end_col]
        start_end_positions = {
            'start_lineno': start_lineno,
            'start_col': start_col,
            'end_lineno': end_lineno,
            'end_col': end_col
        }
        self.generic_visit(node)

        func = node.func
        try:
            if self._is_getattr_call(node):
                # for example "getattr(mylib.a.b, 'const')(x, y)"
                module = node.func.args[0]  # mylib.a.b
                module = astunparse.unparse(module).strip()
                func = node.func.args[1]  # 'const'
                if isinstance(func, ast.Constant):
                    api_name = func.value
                else:  # for example "getattr(mylib.a.b, identifier)(x, y)"
                    return
            elif isinstance(func, ast.Attribute) or isinstance(func, ast.Name):  # "x, y = W.func(a, b, c)"
                module, api_name = self._get_func_name_and_module_from_attribute_or_name(func)
            elif isinstance(func, ast.Subscript):  # for example "x = W.m[0]()"
                # module, api_name = self._get_func_name_and_module_from_attribute_or_name(func.value)
                # api_name += astunparse.unparse(func).strip()[len(astunparse.unparse(func.value))-1:]
                return
            elif isinstance(func, ast.Call):  # for example "x = W.m()()"
                # module, api_name = self._get_func_name_and_module_from_attribute_or_name(func.func)
                # api_name += astunparse.unparse(func).strip()[len(astunparse.unparse(func.func))-1:]
                return
            elif isinstance(func, ast.IfExp):  # for example "(x if None else y)()"
                # module, api_name = '', astunparse.unparse(func).strip()
                return
            elif isinstance(func, ast.Lambda):  # for example "lambda: x()"
                return
            elif isinstance(func, ast.BinOp):  # for example "(ctypes.c_int64 * self._output_ndims[i])()"
                return
            elif isinstance(func, ast.BoolOp):  # for example "(_load or (lambda v: v))(value_)"
                return
            elif func.id == 'getattr':  # don't need to handle getattr() because it is handled in _is_getattr_call()
                return
            self.called_apis.append({
                'api_name': api_name,
                'api_call_prefix': module,
                'api_call_node_start_end_positions': start_end_positions,
                'current_fpath_tuple': self.fpath_tuple
            })
        except Exception as e:
            print(e)
            print(astunparse.unparse(node))
            ipdb.set_trace()


    def _get_func_name_and_module_from_attribute_or_name(self, node: ast.AST):
        if isinstance(node, ast.Attribute):
            module = astunparse.unparse(node.value).strip()
            api_name = node.attr
            return module, api_name
        elif isinstance(node, ast.Name):
            return '', node.id


    def _is_getattr_call(self, base_node: ast.AST) -> bool:
        """
        finds the pattern getattr(mylib, 'const')()
        """
        if not isinstance(base_node, ast.Call):
            return False
        node = base_node.func
        if not isinstance(node, ast.Call):
            return False
        if not (isinstance(node.func, ast.Name) and node.func.id == "getattr"):
            return False
        return True


class APIImportVisitor(ast.NodeVisitor):
    def __init__(self, file_module, fpath_tuple):
        super().__init__()
        self.file_module = file_module
        self.fpath_tuple = fpath_tuple
        self.renamed_api = dict()  # alias of import and func
        self.imported_apis = []

    def visit_Import(self, node: ast.AST):
        '''
        for example "import numpy.array as arr":
        "numpy" is the module stored as value in the "api_path", 
        "array" is the name stored as value in the "remapped", 
        "arr" is the alias stored as key for the "remapped" and "api_path"
            and put "arr" into the "imported_apis"
        '''
        self.generic_visit(node)
        for n in node.names:
            api_name = n.name
            api_as_name = ''
            module = ''
            if '.' in api_name:  # for example "import numpy.array"
                api_name = n.name.split('.')[-1]
                module = '.'.join(n.name.split('.')[:-1])
            if n.asname:
                api_as_name = n.asname
                self.renamed_api[api_as_name] = api_name
            self.imported_apis.append({
                'api_name': api_name,
                'api_path': module,
                'api_as_name': api_as_name,
                'current_fpath_tuple': self.fpath_tuple  # for calculating the similarity between two packages
            })

    def visit_ImportFrom(self, node: ast.AST):
        '''
        for example "from numpy import array as arr":
        "numpy" is the module stored as value in the "api_path", 
        "array" is the name stored as value in the "remapped", 
        "arr" is the alias stored as key for the "remapped" and "api_path"
            and put "arr" into the "imported_apis"
        '''
        self.generic_visit(node)
        module = node.module if node.module else ''
        api_as_name = ''
        if node.level:  # relative import, rebuild module
            if not module and node.level == 1:  # "from . import a" means import from __init__.py
                return
            file_module = self.file_module
            if self.fpath_tuple[-1] == '__init__.py':
                file_module += '.__init__'  # fix the module level of __init__ when doing relative import
            new_module_chain = file_module.split('.')[:-node.level] + [module]
            module = '.'.join([i for i in new_module_chain if i])  # in case module is empty
            
        for n in node.names:
            api_name = n.name
            if n.asname:
                api_as_name = n.asname
                self.renamed_api[api_as_name] = api_name
            self.imported_apis.append({
                'api_name': api_name,
                'api_path': module,
                'api_as_name': api_as_name,
                'current_fpath_tuple': self.fpath_tuple
            })


class APIDefineVisitor(ast.NodeVisitor):
    def __init__(self, fpath_tuple):
        super().__init__()
        self.defined_outer_apis = []
        self.defined_classes = defaultdict(list)
        self.fpath_tuple = fpath_tuple
    
    def store_parent_node(self, root):
        '''
        Remember to first run this function before calling visit
        '''
        for node in ast.walk(root):  # recursive visit
            for child in ast.iter_child_nodes(node):
                child.parent = node
    
    def _get_positon(self, node):
        start_lineno = node.lineno
        start_col = node.col_offset
        end_lineno = node.end_lineno
        end_col = node.end_col_offset
        """
        lineno starts from 1, col_offset starts from 0, and end_offset contains the last symbol
        to extract the segment, use line[start_col:end_col]
        """
        start_end_positions = {
            'start_lineno': start_lineno,
            'start_col': start_col,
            'end_lineno': end_lineno,
            'end_col': end_col
        }
        return start_end_positions
    
    def _build_api_path(self, node):
        api_path = []
        current_node = node
        while hasattr(current_node, 'parent'):
            current_node = current_node.parent
            if isinstance(current_node, ast.ClassDef):
                api_path.insert(0, ('class', current_node.name))
            elif isinstance(current_node, ast.Module):
                break
        return api_path
    
    def _get_func_type(self, node):
        """
        tell whether the function is a class method or an outer method or a local method
        if it is a class method, include the class name
        if it is a local method, return None
        """
        current_node = node
        parent_nodes = []
        while hasattr(current_node, 'parent'):
            current_node = current_node.parent
            if isinstance(current_node, ast.FunctionDef):
                parent_nodes.append(('func', current_node.name))
            elif isinstance(current_node, ast.ClassDef):
                parent_nodes.append(('class', current_node.name))  
        
        if len(parent_nodes) > 1:  # local method, cannot be called by other module
            return ('local', None)
        elif len(parent_nodes) < 1:
            return ('outer', None)
        elif parent_nodes[0][0] == 'func':  # local method  
            return ('local', None)
        elif parent_nodes[0][0] == 'class':  # class method
            return ('class', parent_nodes[0][1])
        else:
            return ('outer', None)
    
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        node_type, class_name = self._get_func_type(node)
        if node_type == 'local':
            return
        docstring = None
        body_index = 0
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
            docstring = node.body[0].value.value
            body_index += 1
        func_node_start_end_positions = self._get_positon(node)
        func_doc_start_end_positions = self._get_positon(node.body[0]) if docstring else None
        func_body_start_end_positions = self._get_positon(node.body[body_index]) if len(node.body) > body_index else None
        
        if node_type == 'outer':
            self.defined_outer_apis.append({
                'api_name': node.name,
                'func_node_start_end_positions': func_node_start_end_positions,
                'func_doc_start_end_positions': func_doc_start_end_positions,
                'func_body_start_end_positions': func_body_start_end_positions,
                'current_fpath_tuple': self.fpath_tuple
            })
        elif node_type == 'class':
            assert class_name
            self.defined_classes[class_name].append({
                'api_name': node.name if node.name != '__init__' else class_name,
                'class_name': class_name,
                'func_node_start_end_positions': func_node_start_end_positions,
                'func_doc_start_end_positions': func_doc_start_end_positions,
                'func_body_start_end_positions': func_body_start_end_positions,
                'current_fpath_tuple': self.fpath_tuple
            })
