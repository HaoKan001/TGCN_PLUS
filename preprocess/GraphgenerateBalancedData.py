import torch
import torch.optim as optim
import time
import itertools
from imblearn.over_sampling import RandomOverSampler
from ParsingSource import *
from Tools import *


now_time = str(int(time.time.time()))
# 创建保存路径
REGENERATE = False
dump_data_path = '/root/autodl-tmp/wgnn/'
os.mkdir(dump_data_path)

# 项目信息
root_path_source = '../data/projects/'
root_path_csv = '../data/csvs/'
package_heads = ['org', 'gnu', 'bsh', 'javax', 'com']

IMBALANCE_PROCESSOR = RandomOverSampler() # RandomOverSampler(), RandomUnderSampler(), None, 'cost'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Analyze source and target projects
path_train_and_test = []
with open('../data/pairs-CPDP.txt', 'r') as file_obj:
    for line in file_obj.readlines():
        line = line.strip('\n')
        line = line.strip(' ')
        path_train_and_test.append(line.split(','))

# Loop each pair of combinations
for path in path_train_and_test:

    # Get file
    path_train_source = root_path_source + path[0]
    path_train_handcraft = root_path_csv + path[0] + '.csv'
    path_test_source = root_path_source + path[1]
    path_test_handcraft = root_path_csv + path[1] + '.csv'

    # Regenerate token or get from dump_data
    print(path[0] + "===" + path[1])
    train_project_name = path_train_source.split('/')[2]
    test_project_name = path_test_source.split('/')[2]
    path_train_and_test_set = dump_data_path + train_project_name + '_to_' + test_project_name
    # If you don't need to regenerate, get it directly from dump_data
    print("开始执行")
    if os.path.exists(path_train_and_test_set) and not REGENERATE:
        print("加载保存的数据")
        obj = load_data(path_train_and_test_set)
        [train_graphs, test_graphs, vocabulary, train_ast, train_hand_craft, train_label, test_ast, test_hand_craft, test_label, vector_len, vocabulary_size] = obj
    else:
        # Get a list of instances of the training and test sets
        print("重新生成")
        train_file_instances = extract_handcraft_instances(path_train_handcraft)

        print("Number of files in train_file_instances :", len(train_file_instances))

        print("First entry in train_file_instances:",
              train_file_instances[0] if len(train_file_instances) > 0 else "No data")

        test_file_instances = extract_handcraft_instances(path_test_handcraft)
        print("Number of files in train_file_instances :", len(test_file_instances))

        # Get tokens
        dict_token_train = parse_source(path_train_source, train_file_instances, package_heads)
        dict_token_test = parse_source(path_test_source, test_file_instances, package_heads)

        # Turn tokens into numbers
        list_dict, vector_len, vocabulary_size = transform_token_to_number([dict_token_train, dict_token_test])

        dict_encoding_train = list_dict[0]
        dict_encoding_test = list_dict[1]

        # Take out data that can be used for training
        train_ast, train_hand_craft, train_label = extract_data(path_train_handcraft, dict_encoding_train)
        test_ast, test_hand_craft, test_label = extract_data(path_test_handcraft, dict_encoding_test)

        # Imbalanced processing
        train_ast, train_hand_craft, train_label, balanced_file_instances = imbalance_process(train_ast, train_hand_craft, train_label, dict_token_train, IMBALANCE_PROCESSOR)

        print("Balanced file instances count:", len(balanced_file_instances))

        # Create graphs from AST with self-loops
        def create_graphs_from_ast(project_root_path, file_instances, package_heads, device):

            dict_graphs, graph_vocabulary = extract_data_and_generate_graphs(project_root_path, file_instances, package_heads)

            vocabulary.update(graph_vocabulary)  # 更新全局词汇表

            graphs = []
            missing_files = []

            # 遍历文件实例列表（包含重复文件）
            for qualified_name, graph in dict_graphs:
                if graph is not None:
                    # 将图移动到指定设备并添加到列表
                    graph = graph.to(device)
                    graphs.append(graph)
                else:
                    # 如果没有生成对应的图，记录缺失的文件
                    missing_files.append(qualified_name)

            if missing_files:
                print("Files without graphs:", missing_files[:10])  # 仅打印前10个缺失的文件

            return graphs, vocabulary

        # Create graphs for training and testing data
        train_graphs, _ = create_graphs_from_ast(path_train_source, balanced_file_instances, package_heads, DEVICE)

        test_graphs, vocabulary = create_graphs_from_ast(path_test_source, dict_token_test.keys(), package_heads, DEVICE)

        # 打印类型映射
        print("Vocabulary (type to index mapping):")
        print(vocabulary)  # 输出每种类型的节点与其索引的映射

        # Saved to dump_data
        obj = [train_graphs, test_graphs, vocabulary, train_ast, train_hand_craft, train_label, test_ast, test_hand_craft, test_label, vector_len, vocabulary_size]
        dump_data(path_train_and_test_set, obj)
        print("保存成功")