import tool.javalang as jl
import os
import tool.javalang.tree as jlt
import numpy as np
import pandas as pd
import networkx as nx
from tool import javalang
import dgl
import torch

types = [jlt.FormalParameter, jlt.BasicType, jlt.PackageDeclaration, jlt.InterfaceDeclaration, jlt.CatchClauseParameter,
         jlt.ClassDeclaration, jlt.MemberReference, jlt.SuperMemberReference, jlt.ConstructorDeclaration, jlt.ReferenceType,
         jlt.MethodDeclaration, jlt.VariableDeclarator, jlt.IfStatement, jlt.WhileStatement, jlt.DoStatement,
         jlt.ForStatement, jlt.AssertStatement, jlt.BreakStatement, jlt.ContinueStatement, jlt.ReturnStatement,
         jlt.ThrowStatement, jlt.SynchronizedStatement, jlt.TryStatement,jlt.SwitchStatement, jlt.BlockStatement,
         jlt.StatementExpression, jlt.TryResource, jlt.CatchClause, jlt.SwitchStatementCase,
         jlt.ForControl, jlt.EnhancedForControl]

types_CPDP = [jlt.FormalParameter, jlt.BasicType, jlt.PackageDeclaration, jlt.InterfaceDeclaration, jlt.CatchClauseParameter,
         jlt.ClassDeclaration, jlt.MethodInvocation, jlt.SuperMethodInvocation,  jlt.SuperMemberReference,
         jlt.ConstructorDeclaration, jlt.ReferenceType, jlt.MethodDeclaration, jlt.VariableDeclarator, jlt.IfStatement,
         jlt.WhileStatement, jlt.DoStatement, jlt.ForStatement, jlt.AssertStatement, jlt.BreakStatement,
         jlt.ContinueStatement, jlt.ReturnStatement, jlt.ThrowStatement, jlt.SynchronizedStatement, jlt.TryStatement,
         jlt.SwitchStatement, jlt.BlockStatement, jlt.StatementExpression, jlt.TryResource, jlt.CatchClause,
         jlt.SwitchStatementCase, jlt.ForControl, jlt.EnhancedForControl, jlt.ClassCreator,
         jlt.TernaryExpression,
         jlt.LambdaExpression,
         jlt.ArrayCreator,
         jlt.TypeParameter,
         jlt.BinaryOperation,
         jlt.ExplicitConstructorInvocation,
         jlt.SuperConstructorInvocation,
         jlt.Assignment,
         jlt.MemberReference,
         jlt.TypeArgument,
         jlt.ConstantDeclaration,
         jlt.EnumDeclaration,
         jlt.ClassReference,
         jlt.LocalVariableDeclaration,
         jlt.Cast,
         jlt.VariableDeclaration,
         jlt.ArraySelector
              ]

features = ['wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom', 'ca', 'ce', 'npm', 'lcom3', 'loc', 'dam', 'moa', 'mfa', 'cam',
            'ic', 'cbm', 'amc', 'max_cc', 'avg_cc']


def extract_data_and_generate_graphs(project_root_path, balanced_file_instances, package_heads, graph_vocabulary = None):
    results = []
    count = 0
    missing_count = 0

    # 遍历 balanced_file_instances 中的每个文件，确保处理重复项
    for qualified_name in balanced_file_instances:
        # 将 qualified_name 转换为文件路径
        matched = False  # 用于检查是否找到匹配的文件
        for dir_path, _, file_names in os.walk(project_root_path):
            # 找到包名的起始位置并构造包名
            index = -1
            for _head in package_heads:
                index = dir_path.find(_head)
                if index >= 0:
                    break
            if index < 0:
                continue

            # 构造包名
            package_name = dir_path[index:].replace(os.sep, '.')

            # 检查每个文件
            for file in file_names:
                if file.endswith('.java') and f"{package_name}.{file}" == qualified_name:
                    # 找到匹配文件，生成图
                    full_path = os.path.join(dir_path, file)
                    graph = parse_java_to_graph(full_path)
                    if graph:
                        dgl_graph = convert_to_dgl(graph, vocabulary = graph_vocabulary)
                        if dgl_graph:
                            results.append((qualified_name, dgl_graph))
                            count += 1
                        else:
                            missing_count += 1
                    else:
                        missing_count += 1
                    matched = True
                    break  # 处理完该文件，退出内层循环

            if matched:
                break  # 找到匹配文件，退出外层循环

        # 如果未找到文件
        if not matched:
            print(f"This file is not in directory structure: {qualified_name}")
            missing_count += 1

    print(f"Total graphs generated: {count}")
    print(f"Missing graphs for files: {missing_count}")
    return results

# 从Java代码生成AST图
def parse_java_to_graph(source_file_path):
    with open(source_file_path, 'rb') as file_obj:
        content = file_obj.read()
    try:
        tree = javalang.parse.parse(content)
    except javalang.parser.JavaSyntaxError as e:
        print(f'JavaSyntaxError in file {source_file_path}: {e.description} at {e.at}')
        return None

    # 使用networkx构建图
    graph = nx.DiGraph()
    node_counter = 0

    # 辅助函数，用于递归遍历AST节点
    def add_nodes_edges(node, parent_id=None):
        nonlocal node_counter

        current_id = node_counter
        node_counter += 1

        # 为图中添加节点，特征包括节点类型
        graph.add_node(current_id, type=type(node).__name__)

        # 如果存在父节点，则创建从父节点到当前节点的边
        if parent_id is not None:
            graph.add_edge(parent_id, current_id)

        # 递归地添加子节点
        if hasattr(node, 'children'):
            for child in node.children:
                if isinstance(child, (list, tuple)):
                    for c in child:
                        if isinstance(c, javalang.tree.Node):
                            add_nodes_edges(c, current_id)
                elif isinstance(child, javalang.tree.Node):
                    add_nodes_edges(child, current_id)

    # 从根节点开始构建图
    add_nodes_edges(tree)
    return graph

def convert_to_dgl(graph, vocabulary=None):

    if not graph:
        return None

    # 设置节点特征，使用类型作为示例特征
    node_types = [graph.nodes[node]['type'] for node in graph.nodes]

    # 使用传入的字典且不修改
    if vocabulary is not None:
        features = [vocabulary.get(node_type, 0) for node_type in node_types]
    else:
        # 如果没有提供字典，默认全0特征（根据需求可修改）
        features = [0] * len(node_types)

    # 转换为张量
    features = torch.tensor(features, dtype=torch.int64)

    # 确保节点数量和特征数量匹配
    assert len(features) == graph.number_of_nodes(), "Mismatch between number of features and number of nodes"

    # 转换为 DGL 图
    dgl_graph = dgl.from_networkx(graph)
    dgl_graph.ndata['feat'] = features

    # 为所有节点添加自环
    dgl_graph = dgl.add_self_loop(dgl_graph)

    # 将图移动到 GPU 上（如果可用）
    if torch.cuda.is_available():
        dgl_graph = dgl_graph.to('cuda')
        dgl_graph.ndata['feat'] = dgl_graph.ndata['feat'].to('cuda')

    # 打印图的节点和边信息
    print(f"Graph info: {dgl_graph.num_nodes()} nodes, {dgl_graph.num_edges()} edges")

    return dgl_graph


#遍历 DataFrame中 file_name 列的每个文件名, 为每个文件名添加 .java 后缀,返回修改后的 DataFrame.
def append_suffix(df):
    for i in range(len(df['file_name'])):
        df.loc[i, 'file_name'] = df.loc[i, 'file_name'] + ".java"
    return df

# 函数返回一个列表，包含 CSV 文件中 file_name 列的所有文件名数据
def extract_handcraft_instances(path):
    handcraft_instances = pd.read_csv(path)
    handcraft_instances = append_suffix(handcraft_instances)
    # 提取名为 'file_name' 的列
    handcraft_instances = np.array(handcraft_instances['file_name'])
    # 将 NumPy 数组转换为 Python 的列表（list）
    handcraft_instances = handcraft_instances.tolist()

    return handcraft_instances


def ast_parse(source_file_path):
    with open(source_file_path, 'rb') as file_obj:
        content = file_obj.read()
        result = []
        tree = []
        try:
            tree = jl.parse.parse(content)
        except jl.parser.JavaSyntaxError as e:
            print('JavaSyntaxError:')
            print(source_file_path)
            print(e.description)
            print(e.at)
        for path, node in tree:
            if isinstance(node, jlt.MethodInvocation) or isinstance(node, jlt.SuperMethodInvocation):
                result.append(str(node.member) + "()")
                continue
            if isinstance(node, jlt.ClassCreator):
                result.append(str(node.type.name))
                continue
            if type(node) in types:
                result.append(str(node))
        return result

# 返回一个文件的所有符合条件的节点
# def ast_parse_CPDP(source_file_path):
#     with open(source_file_path, 'rb') as file_obj:
#         content = file_obj.read()
#         result = []
#         tree = []
#         try:
#             tree = jl.parse.parse(content)
#         except jl.parser.JavaSyntaxError as e:
#             print('JavaSyntaxError:')
#             print(source_file_path)
#             print(e.description)
#             print(e.at)
#         for path, node in tree:
#             if type(node) in types_CPDP:
#                 result.append(str(node))
#                 print(str(node))
#                 print('-----------')
#         return result

def ast_parse_CPDP(source_file_path):
    with open(source_file_path, 'rb') as file_obj:
        content = file_obj.read()
        result = set()
        tree = []
        try:
            tree = jl.parse.parse(content)
        except jl.parser.JavaSyntaxError as e:
            print('JavaSyntaxError:')
            print(source_file_path)
            print(e.description)
            print(e.at)
            # 遍历AST的所有节点，记录类型
        # excluded_types = {"CompilationUnit", "Import"}
        excluded_types = {"CompilationUnit", "PackageDeclaration"}
        for path, node in tree:
            # print(f"实际类型: {type(node)}")  # 输出如 <class 'javalang.tree.EnhancedForControl'>
            # print(f"类型名: {type(node).__name__}")  # 输出如 "EnhancedForControl"
            node_type = type(node).__name__
            if node_type not in excluded_types:
                result.add(node_type)  # 直接追加到列表
                print(f"Node Type: {node_type}")  # 可选：打印类型
                print('-----------')
        # print(result)
        return result


# result = {
#     'com.example.MyClass.java': ['Node1', 'Node2'],
#     'org.example.OtherClass.java': ['NodeA', 'NodeB']
# }
def parse_source(project_root_path, handcraft_file_names, package_heads):
    result = {}
    count = 0
    existed_file_names = []
    for dir_path, dir_names, file_names in os.walk(project_root_path):

        # 如果文件夹下没有文件，直接跳过该文件夹
        if len(file_names) == 0:
            continue

        index = -1
        for _head in package_heads:
            index = int(dir_path.find(_head))
            if index >= 0:
                break
        if index < 0:
            continue

        package_name = dir_path[index:]
        package_name = package_name.replace(os.sep, '.')

        for file in file_names:
            if file.endswith('java'):
                if str(package_name + "." + str(file)) not in handcraft_file_names:
                    continue

                ast_result  = ast_parse_CPDP(str(os.path.join(dir_path, file)))

                if ast_result:  # ast_result != []
                    result[package_name + "." + str(file)] = ast_result
                    existed_file_names.append(str(package_name + "." + str(file)))
                    count += 1

    for handcraft_file_name in handcraft_file_names:
        handcraft_file_name.replace('.java', '')
        if handcraft_file_name not in existed_file_names:
            print('This file is not in csv list:' + handcraft_file_name)

    print("data size : " + str(count))
    return result


# def padding_vector(vector, size):
#     if len(vector) == size:
#         return vector
#     padding = np.zeros((1, size - len(vector)))
#     padding = list(np.squeeze(padding))
#     vector += map(int, padding)
#     return vector

def padding_vector(vector, size):
    if len(vector) >= size:
        return vector[:size]

    pad_length = size - len(vector)
    padding = np.zeros(pad_length).tolist()  # 直接转列表（避免维度问题）
    vector += padding
    return vector


def padding_all(dict_token, size):
    result = {}
    for key, vector in dict_token.items():
        pv = padding_vector(vector, size)
        result[key] = pv
    return result


def max_length(d):
    max_len = 0
    for value in d.values():
        if max_len < len(value):
            max_len = len(value)
    return max_len


# result = {
#     'com.example.MyClass.java': ['1', '2'],
#     'org.example.OtherClass.java': ['3', '4']
# }
def transform_token_to_number(list_dict_token):
    frequence = {}
    for _dict_token in list_dict_token:
        for _token_vector in _dict_token.values():
            for _token in _token_vector:
                if frequence.__contains__(_token):
                    frequence[_token] = frequence[_token] + 1
                else:
                    frequence[_token] = 1

    vocabulary = {}  # 用来学习每个token的数字表示的映射表    token:number
    result = []
    count = 0
    max_len = 0
    for dict_token in list_dict_token:
        _dict_encode = {}
        for file_name, token_vector in dict_token.items():
            vector = []
            for v in token_vector:
                if frequence[v] < 3:
                    continue

                if vocabulary.__contains__(v):
                    vector.append(vocabulary.get(v))
                else:
                    count = count + 1
                    vector.append(count)
                    vocabulary[v] = count
            if len(vector) > max_len:
                max_len = len(vector)
            _dict_encode[file_name] = vector
        result.append(_dict_encode)

    for i in range(len(result)):
        result[i] = padding_all(result[i], max_len)
    return result, max_len, len(vocabulary)


def extract_data(path_handcraft_file, dict_encoding_vector):
    # 找到对应文件的bug值，如果大于1返回1，如果是0或1不变，也就是说修改bug列的值，变成只有0或1
    def extract_label(df, file_name):
        row = df[df.file_name == file_name]['bug']
        row = np.array(row).tolist()
        if row[0] > 1:
            row[0] = 1
        return row

    def extract_feature(df, file_name):
        row = df[df.file_name == file_name][features]
        row = np.array(row).tolist()
        row = np.squeeze(row)
        row = list(row)
        return row

    ast_x_data = []
    hand_x_data = []
    label_data = []
    raw_handcraft = pd.read_csv(path_handcraft_file)
    raw_handcraft = append_suffix(raw_handcraft)
    for key, value in dict_encoding_vector.items():
        ast_x_data.append(value)
        # key就是一个文件名列表，就是手工文件在项目里能够匹配的那些文件的列表
        hand_x_data.append(extract_feature(raw_handcraft, key))
        label_data.append(extract_label(raw_handcraft, key))
    ast_x_data = np.array(ast_x_data)
    hand_x_data = np.array(hand_x_data)
    label_data = np.array(label_data)

    return ast_x_data, hand_x_data, label_data

def extract_label(path_handcraft_file, dict_encoding_vector):
    # 找到对应文件的bug值，如果大于1返回1，如果是0或1不变，也就是说修改bug列的值，变成只有0或1
    def extract_label(df, file_name):
        row = df[df.file_name == file_name]['bug']
        row = np.array(row).tolist()
        if row[0] > 1:
            row[0] = 1
        return row

    def extract_feature(df, file_name):
        row = df[df.file_name == file_name][features]
        row = np.array(row).tolist()
        row = np.squeeze(row)
        row = list(row)
        return row

    label_data = []
    raw_handcraft = pd.read_csv(path_handcraft_file)
    raw_handcraft = append_suffix(raw_handcraft)
    for key, value in dict_encoding_vector.items():
        # key就是一个文件名列表，就是手工文件在项目里能够匹配的那些文件的列表
        label_data.append(extract_label(raw_handcraft, key))
    label_data = np.array(label_data)

    return label_data


    # # 打印图的基本结构信息
    # print("Graph basic structure:")
    # print(dgl_graph)
    #
    # # 打印所有节点的ID
    # print("\nNode IDs:")
    # print(dgl_graph.nodes())
    #
    # # 打印图的边列表 (src -> dst)
    # print("\nEdge list (src -> dst):")
    # print(dgl_graph.edges())
    #
    # # 打印节点特征
    # print("Node features (ndata):")
    # print(dgl_graph.ndata['feat'])  # 输出所有节点的整数特征
    #
    # # 打印类型映射
    # print("Vocabulary (type to index mapping):")
    # print(vocabulary)  # 输出每种类型的节点与其索引的映射
# types_CPDP = [
#     # ================ 编译单元与顶层声明 ================
#     # jlt.CompilationUnit,
#     jlt.PackageDeclaration,
#     # jlt.Import,
#
#     # ================ 类型声明 ================
#     jlt.ClassDeclaration,
#     jlt.InterfaceDeclaration,
#     jlt.EnumDeclaration,
#     jlt.AnnotationDeclaration,
#     jlt.TypeDeclaration,
#
#     # ================ 成员声明 ================
#     jlt.MethodDeclaration,
#     jlt.FieldDeclaration,
#     jlt.ConstructorDeclaration,
#     jlt.ConstantDeclaration,
#     jlt.AnnotationMethod,
#
#     # ================ 变量与参数 ================
#     jlt.VariableDeclaration,
#     jlt.LocalVariableDeclaration,
#     jlt.VariableDeclarator,
#     jlt.FormalParameter,
#     jlt.InferredFormalParameter,
#
#     # ================ 语句 ================
#     jlt.IfStatement,
#     jlt.ForStatement,
#     jlt.WhileStatement,
#     jlt.DoStatement,
#     jlt.SwitchStatement,
#     jlt.TryStatement,
#     jlt.BlockStatement,
#     jlt.ReturnStatement,
#     jlt.ThrowStatement,
#     jlt.BreakStatement,
#     jlt.ContinueStatement,
#     jlt.AssertStatement,
#     jlt.SynchronizedStatement,
#     jlt.CatchClause,
#     jlt.TryResource,
#     jlt.SwitchStatementCase,
#     jlt.ForControl,
#     jlt.EnhancedForControl,
#     jlt.CatchClauseParameter,
#
#     # ================ 表达式 ================
#     jlt.Assignment,
#     jlt.TernaryExpression,
#     jlt.BinaryOperation,
#     jlt.Cast,
#     jlt.MethodReference,
#     jlt.LambdaExpression,
#     jlt.ArraySelector,
#     jlt.StatementExpression,
#
#     # ================ 类型与泛型 ================
#     jlt.BasicType,
#     jlt.ReferenceType,
#     jlt.TypeArgument,
#     jlt.TypeParameter,
#
#     # ================ 注解 ================
#     # jlt.Annotation,
#     # jlt.ElementValuePair,
#     # jlt.ElementArrayValue,
#
#     # ================ 对象创建与调用 ================
#     jlt.ClassCreator,
#     jlt.ArrayCreator,
#     jlt.InnerClassCreator,
#     jlt.MethodInvocation,
#     jlt.SuperMethodInvocation,
#     jlt.ExplicitConstructorInvocation,
#     jlt.SuperConstructorInvocation,
#
#     # ================ 基础表达式与字面量 ================
#     jlt.Literal,
#     jlt.This,
#     jlt.MemberReference,
#     jlt.SuperMemberReference,
#     jlt.ClassReference,
#     jlt.VoidClassReference,
#
#     # ================ 其他辅助节点 ================
#     # jlt.EnumBody,
#     # jlt.EnumConstantDeclaration,
#     # jlt.ArrayInitializer,
#     # jlt.Documented,
#     # jlt.Declaration,
#     # jlt.Expression,
#     # jlt.Primary,
# ]
# types_CPDP = [jlt.FormalParameter, jlt.BasicType, jlt.PackageDeclaration, jlt.InterfaceDeclaration, jlt.CatchClauseParameter,
#          jlt.ClassDeclaration, jlt.MethodInvocation, jlt.SuperMethodInvocation, jlt.MemberReference, jlt.SuperMemberReference,
#          jlt.ConstructorDeclaration, jlt.ReferenceType, jlt.MethodDeclaration, jlt.VariableDeclarator, jlt.IfStatement,
#          jlt.WhileStatement, jlt.DoStatement, jlt.ForStatement, jlt.AssertStatement, jlt.BreakStatement,
#          jlt.ContinueStatement, jlt.ReturnStatement, jlt.ThrowStatement, jlt.SynchronizedStatement, jlt.TryStatement,
#          jlt.SwitchStatement, jlt.BlockStatement, jlt.StatementExpression, jlt.TryResource, jlt.CatchClause,
#          jlt.SwitchStatementCase, jlt.ForControl, jlt.EnhancedForControl, jlt.ClassCreator,
#          jlt.TernaryExpression,
#          jlt.LambdaExpression,
#          jlt.ArrayCreator,
#          jlt.TypeParameter,
#          jlt.BinaryOperation,
#          jlt.ExplicitConstructorInvocation,
#          jlt.SuperConstructorInvocation,
#          jlt.InferredFormalParameter,
#          jlt.Assignment
#               ]