import datetime
import itertools
from imblearn.over_sampling import RandomOverSampler
from ParsingSource import *
from Tools import *
from tool import javalang
import tool.javalang.tree as jlt
import tool.javalang as jl
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
import random
import numpy as np
from collections import Counter

# Start Time
start_time = datetime.datetime.now()
start_time_str = start_time.strftime('%Y-%m-%d_%H.%M.%S')

def random_oversample(file_node_types, labels):

    assert len(file_node_types) == len(labels), "特征与标签数量不一致"

    keys = list(file_node_types.keys())
    features = list(file_node_types.values())
    labels = np.array(labels)

    # 统计每个类别的样本数
    class_counts = Counter(labels)
    max_count = max(class_counts.values())

    # 新数据容器
    new_keys = keys.copy()
    new_features = features.copy()
    new_labels = labels.tolist()

    # 记录重复键的次数，确保键唯一
    key_counts = Counter()

    for class_label, count in class_counts.items():
        if count < max_count:
            needed = max_count - count
            class_indices = np.where(labels == class_label)[0]

            for _ in range(needed):
                selected_idx = np.random.choice(class_indices)

                original_key = keys[selected_idx]

                # 生成唯一键名
                key_counts[original_key] += 1
                new_key = f"{original_key}_ovs_{key_counts[original_key]}"

                # 追加数据
                new_keys.append(new_key)
                new_features.append(features[selected_idx])
                new_labels.append(class_label)

    # 最终构建字典确保顺序对齐
    new_file_node_types = dict(zip(new_keys, new_features))

    return new_file_node_types, new_labels

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
            print('This file in csv list is not in project:' + handcraft_file_name)

    print("data size : " + str(count))
    return result

def ast_parse_CPDP(source_file_path):
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
            # 遍历AST的所有节点，记录类型
        excluded_types = {"CompilationUnit", "PackageDeclaration"}

        for path, node in tree:
            # print(f"实际类型: {type(node)}")  # 输出如 <class 'javalang.tree.EnhancedForControl'>
            # print(f"类型名: {type(node).__name__}")  # 输出如 "EnhancedForControl"
            node_type = type(node).__name__
            if node_type not in excluded_types:
                result.append(node_type)  # 直接追加到列表
                print(f"Node Type: {node_type}")  # 可选：打印类型
                print('-----------')
        print(result)
        return result

# 创建保存路径
REGENERATE = False
dump_data_path = '/root/autodl-tmp/'

# 项目信息
root_path_source = '../data/projects/'
root_path_csv = '../data/csvs/'
package_heads = ['org', 'gnu', 'bsh', 'javax', 'com']

IMBALANCE_PROCESSOR = RandomOverSampler() # RandomOverSampler(), RandomUnderSampler(), None, 'cost'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Analyze projects
path_projects = []
with open('../data/pairs-pre.txt', 'r') as file_obj:
    for line in file_obj:
        stripped_line = line.strip()  # 移除换行符和空白
        if stripped_line:  # 忽略空行
            path_projects.append(stripped_line)


for project in path_projects:
    # 生成项目相关路径
    path_source = root_path_source + project
    path_handcraft = root_path_csv + project + '.csv'
    project_name = path_source.rstrip('/').split('/')[-1]
    # 打印三个关键变量
    print("\n=== 当前项目信息 ===")
    print(f"1. path_source: {path_source}")
    print(f"2. path_handcraft: {path_handcraft}")
    print(f"3. project_name: {project_name}")

    path_set = dump_data_path + project_name

    # If you don't need to regenerate, get it directly from dump_data
    print("开始执行")
    if os.path.exists(path_set) and not REGENERATE:
        print("加载保存的数据")
        obj = load_data(path_set)
        [train_graphs, test_graphs, train_ast, train_hand_craft, train_label, test_ast, test_hand_craft, test_label, vector_len, vocabulary_size] = obj
    else:
        # Get a list of instances of the training and test sets
        print("重新生成")
        file_instances = extract_handcraft_instances(path_handcraft)

        file_node_type = parse_source(path_source, file_instances, package_heads)

        all_node_types = set()
        for node_list in file_node_type.values():
            all_node_types.update(node_list)

        sorted_node_types = sorted(all_node_types)

        print(f"\n项目 {project_name} 的节点类型（共 {len(all_node_types)} 种）:")
        print(sorted_node_types)  # 直接打印集合，自动显示为 {...} 格式

        label = extract_label(path_handcraft, file_node_type).flatten()

        # 打印原始数据分布
        print("【过采样前】")
        print(f"特征数量: {len(file_node_type)}")
        print(f"标签分布: {Counter(label)}")
        print(f"0类数量: {list(label).count(0)}")
        print(f"1类数量: {list(label).count(1)}\n")

        file_node_types, labels = random_oversample(file_node_type, label)

        # 打印过采样结果
        print("【过采样后】")
        print(f"特征数量: {len(file_node_types)}")
        print(f"标签分布: {Counter(labels)}")
        print(f"0类数量: {labels.count(0)}")
        print(f"1类数量: {labels.count(1)}")

        # 1. 统计每个文件的节点类型频次
        file_features = {
            file: Counter(node_types)
            for file, node_types in file_node_types.items()
        }

        # 2. 转换为DataFrame（行是文件，列是节点类型，值是频次）
        df = pd.DataFrame.from_dict(file_features, orient='index').fillna(0)

        df.to_csv(f'/root/autodl-tmp/{project_name}_features_matrix.csv')

        min_total_freq = 3
        min_class_freq = 2

        df = df.loc[:, df.sum() >= min_total_freq]
        y = pd.Series(labels, index=df.index)

        valid_nodes = df.columns[
            (df[y == 1].sum() >= min_class_freq) |
            (df[y == 0].sum() >= min_class_freq)
            ]

        df_filtered = df[valid_nodes]

        # 二元化特征（存在=1，不存在=0）
        df_binary = df_filtered.applymap(lambda x: 1 if x > 0 else 0)

        # ------------------------------
        # 3. Fisher精确检验 + 多重比较校正
        # ------------------------------
        no_significant_nodes = []
        significant_nodes = []
        alpha = 0.05  # 原始显著性水平

        test_count = 0
        for node_type in df_filtered.columns:
            contingency_table = pd.crosstab(df_binary[node_type], y)
            if contingency_table.shape == (2, 2):
                test_count += 1
        adjusted_alpha = alpha / test_count  # 按实际检验次数校正 # Bonferroni校正

        for node_type in df_filtered.columns:
            # 构建2x2列联表
            contingency_table = pd.crosstab(df_binary[node_type], y)

            # # 如果是WhileStatement，打印列联表
            # if node_type == "ClassDeclaration":
            #     print(f"WhileStatement 的列联表:\n{contingency_table}\n")

            # 确保列联表是2x2（否则跳过）
            if contingency_table.shape != (2, 2):
                print(f"跳过节点 {node_type}（列联表非2x2）")
                continue

            # 执行Fisher精确检验（使用蒙特卡洛模拟加速，适用于大数据）
            odds_ratio, p_value = fisher_exact(
                contingency_table,
                alternative='greater'  # 检测正相关
                # simulate_pval=True,      # 大数据时启用蒙特卡洛模拟
                # replicates=10000        # 蒙特卡洛模拟次数
            )
            # if odds_ratio > 1:
            if p_value < adjusted_alpha or odds_ratio >1 :
                significant_nodes.append(node_type)
            else:
                no_significant_nodes.append(node_type)

        # alpha = 0.05
        # p_values = []
        # odds_ratios = []
        # valid_nodes = []
        #
        # # 批量计算列联表和统计量
        # for node_type in df_filtered.columns:
        #     contingency_table = pd.crosstab(df_binary[node_type], y)
        #     if contingency_table.shape != (2, 2):
        #         continue
        #     odds_ratio, p_value = fisher_exact(contingency_table, alternative='greater')
        #
        #     valid_nodes.append(node_type)
        #     p_values.append(p_value)
        #     odds_ratios.append(odds_ratio)
        #
        # # FDR校正
        # rejects, corrected_pvals, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
        #
        # # 严格显著性判断 (p值显著且odds_ratio>1)
        # significant_nodes = [
        #     node for node, reject, oratio in zip(valid_nodes, rejects, odds_ratios)
        #     # if reject or oratio > 1
        #     if oratio > 1
        # ]
        #
        # no_significant_nodes = [
        #     node for node in valid_nodes if node not in significant_nodes
        # ]
        #
        # # 显示结果
        print(f"显著节点 ({len(significant_nodes)}个): {significant_nodes}")
        print(f"非显著节点 ({len(no_significant_nodes)}个): {no_significant_nodes}")
        # ------------------------------
        # 4. 结果输出
        # ------------------------------
        # print("\n========== 分析结果 ==========")
        # print(f"原始特征数: {len(df.columns)}")
        # print(f"过滤后特征数: {len(df_filtered.columns)}")
        # print(f"显著相关的节点类型（校正后 α={adjusted_alpha:.4f}）: {significant_nodes}")
        # print(f"不显著的节点类型（校正后 α={adjusted_alpha:.4f}）: {no_significant_nodes}")

        _, vocabulary = load_data(os.path.join('/root/autodl-tmp/', 'global_vocabulary.pkl'))
        indices = [vocabulary[node] for node in sorted_node_types]

        dump_data(os.path.join(dump_data_path, f'{project_name}_indices01.pkl'), indices)

        print(indices)

        # 输出所有节点的检验统计量
        # stats_list = []
        # for node_type in df_filtered.columns:
        #     contingency_table = pd.crosstab(df_binary[node_type], y)
        #     if contingency_table.shape != (2, 2):
        #         continue
        #     odds_ratio, p_value = fisher_exact(contingency_table, alternative='greater')
        #     stats_list.append({
        #         "NodeType": node_type,
        #         "OddsRatio": odds_ratio,
        #         "PValue": p_value,
        #         "Significant": p_value < adjusted_alpha
        #     })
        #
        # stats_df = pd.DataFrame(stats_list).sort_values("PValue")
        # print("\n所有节点检验统计量:")
        # stats_df.to_csv(f'/root/autodl-tmp/{project_name}_stats_df.csv')
        # print(stats_df)
        # stats_list = []
        #
        # for node_type in df_filtered.columns:
        #     contingency_table = pd.crosstab(df_binary[node_type], y)
        #
        #     if contingency_table.shape != (2, 2):
        #         continue
        #
        #     # 提取频数
        #     a = contingency_table.loc[1, 1]
        #     b = contingency_table.loc[1, 0]
        #     c = contingency_table.loc[0, 1]
        #     d = contingency_table.loc[0, 0]
        #
        #     corrected = False
        #
        #     # 检查是否需要平滑（避免 OR=inf 或 OR=0）
        #     if 0 in [a, b, c, d]:
        #         corrected = True
        #         a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        #         odds_ratio = (a * d) / (b * c)
        #         # Fisher精确检验不能用校正后的值，只返回 NaN 或空
        #         p_value = np.nan
        #     else:
        #         # 直接使用原始列联表
        #         odds_ratio, p_value = fisher_exact(contingency_table, alternative='greater')
        #
        #     stats_list.append({
        #         "NodeType": node_type,
        #         "OddsRatio": odds_ratio,
        #         "PValue": p_value,
        #         "Corrected": corrected,
        #         "Significant": (p_value < adjusted_alpha) if not np.isnan(p_value) else False
        #     })
        #
        # # 构建 DataFrame，按 p 值排序（NaN 放后面）
        # stats_df = pd.DataFrame(stats_list).sort_values("PValue", na_position='last')
        #
        # # 输出结果
        # print("\n所有节点检验统计量:")
        # stats_df.to_csv(f'/root/autodl-tmp/{project_name}_stats_df.csv', index=False)
        # print(stats_df)