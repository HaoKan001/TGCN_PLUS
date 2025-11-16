import os
import numpy as np
import pickle


def round_self(array):
    result = list()
    for a in array:
        if a > 0.5:
            result.append(1)
        else:
            result.append(0)
    return result


def generate_dir_recursive(path):
    import os
    path = path.strip()
    path = path.rstrip('/')
    phases = path.split('/')
    path = ''
    for i in range(len(phases)):
        path = path + phases[i] + '/'
        if not os.path.exists(path):
            os.mkdir(path)


def dump_data(path, obj):
    with open(path, 'wb') as file_obj:
        pickle.dump(obj, file_obj)


def load_data(path):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as file_obj:
        return pickle.load(file_obj)


def generate_weight_for_instance(label):
    a = 0
    b = 0
    label = np.array(label)
    label = label.reshape(-1)
    for i in label:
        if i == 0:
            a = a + 1
        else:
            b = b + 1
    s = a + b
    a = a * 1.0 / s
    b = b * 1.0 / s
    result = []
    for i in label:
        if i == 0:
            result.append(b)
        else:
            result.append(a)
    return np.array(result)


def generate_weight_for_class(label):
    a = 0
    b = 0
    label = np.array(label)
    label = label.reshape(-1)
    for i in label:
        if i == 0:
            a = a + 1
        else:
            b = b + 1
    s = a + b
    a = a * 1.0 / s
    b = b * 1.0 / s
    result = [b, a]
    return np.array(result)

# TCNN
# def imbalance_process(x_ast, x_handcraft, label, processor):
#     if processor is None:
#         return x_ast, x_handcraft, label
#
#     _len_ast = len(x_ast[0])
#     _len_label = len(label)
#
#     _x = np.hstack((x_ast, x_handcraft))
#     _y = label.reshape(_len_label)
#     _x_resampled, _y_resampled = processor.fit_resample(_x, _y)
#     _ast, _handcraft = np.split(_x_resampled, [_len_ast], axis=1)
#     _label = _y_resampled.reshape((len(_y_resampled), 1))
#
#     print("after imbalance process:" + str(len(_y_resampled)))
#     _count = 0
#     for i in _y_resampled:
#         if i == 0:
#             _count += 1
#     print("0 count is:" + str(_count))
#
#     return _ast, _handcraft, _label

# TGNN
def imbalance_process(x_ast, x_handcraft, label, file_instances, processor):
    # 检查 processor 是否为 None
    if processor is None:
        return x_ast, x_handcraft, label, file_instances

        # 提取 file_instances 的文件名
    file_names = [file_instance[0] if isinstance(file_instance, tuple) else file_instance for file_instance in
                      file_instances]

    # 检查 x_ast 是否为空或格式不正确
    if len(x_ast) == 0 or len(x_ast[0]) == 0:
        print("Error: x_ast is empty or not properly formatted")
        return x_ast, x_handcraft, label, file_instances


    # 获取 x_ast 中的特征长度和标签的长度
    _len_ast = len(x_ast[0])
    _len_label = len(label)

    # 合并 x_ast 和 x_handcraft 以便进行采样
    _x = np.hstack((x_ast, x_handcraft))
    _y = label.reshape(_len_label)

    # 使用 processor 进行采样，以实现数据平衡
    _x_resampled, _y_resampled = processor.fit_resample(_x, _y)

    # 分割处理后的数据，将 _x_resampled 拆分为 AST 特征和手工特征
    _ast, _handcraft = np.split(_x_resampled, [_len_ast], axis=1)
    _label = _y_resampled.reshape((len(_y_resampled), 1))

    # 使用重新采样的索引生成新的文件列表，不去重重复文件
    balanced_file_indices = processor.sample_indices_
    balanced_file_instances = [file_names[idx] for idx in balanced_file_indices]

    # 输出重新采样后的样本总数以及标签为0的样本数
    print("after imbalance process: " + str(len(_y_resampled)))
    _count = sum(1 for i in _y_resampled if i == 0)
    print("0 count is: " + str(_count))

    # 返回处理后的数据和文件实例列表
    return _ast, _handcraft, _label, balanced_file_instances



def count_binary(label):
    _count = 0
    for i in range(len(label)):
        if label[i][0] == 0:
            _count += 1
    print("0:" + str(_count))
    print("1:" + str(len(label) - _count))
