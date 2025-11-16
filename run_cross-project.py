import math
import torch
import torch.optim as optim
import datetime
import itertools
from imblearn.over_sampling import RandomOverSampler
import MMD
from TGAT_Test import WGNN_test
from ParsingSource import *
from Tools import *
import dgl
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
import torch.nn as nn
# from models.TACGNN import CombinedModel
import pandas as pd
import os
# from models.gat import AdversarialGAT
from models.gcn import AdversarialGAT
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy
from torch.optim.lr_scheduler import CosineAnnealingLR
def generate_mask(g, critical_nodes):
    """生成动态掩码：节点类型在 critical_nodes 中的位置为1，否则为0"""
    node_types = g.ndata['feat']  # 假设节点类型字段是 'type'
    critical_nodes_tensor = torch.tensor(critical_nodes, device=node_types.device)
    mask = torch.isin(node_types, critical_nodes_tensor).float()
    # mask = torch.isin(node_types, torch.tensor(critical_nodes)).float()  # [num_nodes]
    return mask.to(DEVICE)

def domain_diagnostics(domain_logits, domain_labels, step=None, show_plot=False):
    """
    domain_logits: [batch_size, 2] — 来自 domain_clf 的原始输出（未 softmax）
    domain_labels: [batch_size] — 0: 源域, 1: 目标域
    """
    with torch.no_grad():
        probs = F.softmax(domain_logits, dim=1)  # [B, 2]
        avg_probs = probs.mean(dim=0).cpu().numpy()  # [2,]
        pred_labels = probs.argmax(dim=1)

        acc = (pred_labels == domain_labels).float().mean().item()

        print(f"📊 Domain Acc: {acc:.4f} | Prob (Src, Tgt): {avg_probs[0]:.4f}, {avg_probs[1]:.4f}")

        # if show_plot:
        #     plt.bar(['Source', 'Target'], avg_probs, color=['blue', 'orange'])
        #     plt.title(f'Domain prediction prob at step {step}')
        #     plt.ylim(0, 1)
        #     plt.ylabel('Softmax Avg')
            # plt.show()

# Initial super-parameter of network
init_gnn_params = {'EMBED_DIM': 32, 'N_HIDDEN_NODE': 100, 'N_EPOCH': 1, 'LEARNING_RATE': 5e-5,
                   'MOMEMTUN': 0.9, 'L2_WEIGHT': 0.005, 'DROPOUT': 0.5, 'STRIDE': 1, 'PADDING': 0, 'POOL_SIZE': 2,'DICT_SIZE':0, 'TOKEN_SIZE': 0}

# Adjustable parameter
REGENERATE = False
# dump_data_path = 'data/balanced_dump_data_cross-project2/'
# dump_data_path = '/root/autodl-tmp/zihuan/'
dump_data_path = '/root/autodl-tmp/GAT/'

# Fixed parameter
IMBALANCE_PROCESSOR = RandomOverSampler()  # RandomOverSampler(), RandomUnderSampler(), None, 'cost'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')
root_path_source = 'data/projects/'
root_path_csv = 'data/csvs/'
package_heads = ['org', 'gnu', 'bsh', 'javax', 'com']

# Start Time
start_time = datetime.datetime.now()
start_time_str = start_time.strftime('%Y-%m-%d_%H.%M.%S')

WGNN = {'acc': [], 'auc': [], 'f1': [], 'mcc': [], 'gmean': [], 'precision': [], 'recall': []}

# Get a list of source and target projects
path_train_and_test = []
with open('data/pairs-one.txt', 'r') as file_obj:
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
        [train_graphs, test_graphs, train_ast, train_hand_craft, train_label, test_ast, test_hand_craft, test_label, vector_len, vocabulary_size] = obj
    else:
        # Get a list of instances of the training and test sets
        print("重新生成")
        train_file_instances = extract_handcraft_instances(path_train_handcraft)
        test_file_instances = extract_handcraft_instances(path_test_handcraft)

        # Get tokens
        dict_token_train = parse_source(path_train_source, train_file_instances, package_heads)
        dict_token_test = parse_source(path_test_source, test_file_instances, package_heads)

        # Turn tokens into numbers
        list_dict, vector_len, vocabulary_size1 = transform_token_to_number([dict_token_train, dict_token_test])

        vocabulary_size, vocabulary = load_data(os.path.join('/root/autodl-tmp/', 'global_vocabulary.pkl'))

        dict_encoding_train = list_dict[0]
        dict_encoding_test = list_dict[1]

        # Take out data that can be used for training
        train_ast, train_hand_craft, train_label = extract_data(path_train_handcraft, dict_encoding_train)
        test_ast, test_hand_craft, test_label = extract_data(path_test_handcraft, dict_encoding_test)

        # Imbalanced processing
        train_ast, train_hand_craft, train_label, balanced_file_instances = imbalance_process(train_ast, train_hand_craft, train_label, dict_token_train, IMBALANCE_PROCESSOR)

        print("Balanced file instances count:", len(balanced_file_instances))

        # Create graphs from AST with self-loops
        def create_graphs_from_ast(project_root_path, file_instances, package_heads, device, vocabulary=None):
            dict_graphs = extract_data_and_generate_graphs(project_root_path, file_instances, package_heads, graph_vocabulary = vocabulary)

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

            return graphs

        # Create graphs for training and testing data
        train_graphs = create_graphs_from_ast(path_train_source, balanced_file_instances, package_heads, DEVICE, vocabulary = vocabulary)

        test_graphs = create_graphs_from_ast(path_test_source, dict_token_test.keys(), package_heads, DEVICE, vocabulary = vocabulary)

        # Saved to dump_data
        obj = [train_graphs, test_graphs, train_ast, train_hand_craft, train_label, test_ast, test_hand_craft, test_label, vector_len, vocabulary_size]
        dump_data(path_train_and_test_set, obj)
        print("保存成功")

    # ZScore
    train_hand_craft = (train_hand_craft - np.mean(train_hand_craft, axis=0)) / np.std(train_hand_craft, axis=0)
    test_hand_craft = (test_hand_craft - np.mean(test_hand_craft, axis=0)) / np.std(test_hand_craft, axis=0)

    train_src = load_data(os.path.join('/root/autodl-tmp/', f'{train_project_name}_indices.pkl'))
    test_tar = load_data(os.path.join('/root/autodl-tmp/', f'{test_project_name}_indices01.pkl'))
    # print(train_src)
    # print(test_tar)
    intersection_set = set(train_src) & set(test_tar)

    # critical_nodes = list(intersection_set)

    # 原来研究使用的节点类型
    # critical_nodes = [7, 9, 11, 12, 14, 15, 16, 17, 20, 21, 22, 25, 31, 32, 33, 34, 36, 39, 40, 41, 42, 43, 45, 47, 48, 49, 50, 51, 54, 55, 59, 60]

    # 仅使用源/目标项目节点类型
    critical_nodes = train_src

    # print("节点集合：", critical_nodes)

    class ASTDataset(DGLDataset):
        def __init__(self, graphs, ast, labels, device):
            self.graphs = [g.to(device) for g in graphs]
            self.ast = torch.tensor(ast).to(device)
            self.labels = torch.tensor(labels).to(device)
            super().__init__(name='ast_dataset')

        def process(self):
            pass

        def __getitem__(self, idx):
            return self.graphs[idx], self.ast[idx], self.labels[idx]

        def __len__(self):
            return len(self.graphs)

    # Define the dataset
    train_dataset = ASTDataset(train_graphs, train_ast, train_label,DEVICE)
    test_dataset = ASTDataset(test_graphs, test_ast, test_label,DEVICE)

    train_loader = GraphDataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
    test_loader = GraphDataLoader(test_dataset, batch_size=16 , shuffle=True, drop_last=False)

    # Select nn parameters
    gnn_params = init_gnn_params.copy()
    gnn_params['DICT_SIZE'] = vocabulary_size + 1

    # 每 step 的记录
    loss_history = {
        'total': [],
        'defect': [],
        'mmd': []
    }

    # 每 epoch 的平均记录
    epoch_loss_history = {
        'total': [],
        'defect': [],
        'mmd': []
    }
    # ------------------ GNN training begins ------------------
    GNN_acc, GNN_auc, GNN_f1, GNN_mcc, GNN_gmean, GNN_precision, GNN_recall = [], [], [], [], [], [], []

    model = AdversarialGAT(vocab_size = vocabulary_size + 1)
    model.to(DEVICE)

    best_defect_loss = float('inf')
    best_model_state = None
    no_improve_epochs = 0
    patience = 1
    # stop_on_domain_acc = False  # 可切换是否用 domain acc early stop

    optimizer = optim.Adam(model.parameters(), lr=gnn_params['LEARNING_RATE'], betas=(0.9, 0.999), eps=1e-8, weight_decay=gnn_params['L2_WEIGHT'], amsgrad=False)

    for epoch in range(gnn_params['N_EPOCH']):
        total_defect_loss = 0
        total_loss_train = 0
        total_mmd_loss = 0
        # # 余弦退火调度（alpha ∈ [0, 1]）
        alpha = 0.5 * (1 + torch.cos(torch.tensor(epoch / gnn_params['N_EPOCH'] * 3.1416)))
        lambda_adv = 2.0 * (epoch / gnn_params['N_EPOCH']) # 动态对抗权重

        num_steps_per_epoch = len(train_loader)  # 或 len(train_loader) max(len(train_loader), len(test_loader))

        # 源领域数据
        source_iter = iter(train_loader)
        target_iter = iter(test_loader)
        for step in range(num_steps_per_epoch):
            try:
                batch_graph, batch_ast_x, batch_y = next(source_iter)
            except StopIteration:
                source_iter = iter(train_loader)
                batch_graph, batch_ast_x, batch_y = next(source_iter)

            try:
                tgt_graph, _, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(test_loader)
                tgt_graph, _, _ = next(target_iter)

            batch_graph = batch_graph.to(DEVICE)
            batch_graph.ndata['feat'] = batch_graph.ndata['feat'].float().to(DEVICE)
            batch_ast_x = batch_ast_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)


            batch_domain_labels = torch.zeros(batch_graph.batch_size, dtype=torch.long).to(DEVICE)

            # 目标领域数据
            tgt_graph = tgt_graph.to(DEVICE)
            tgt_graph.ndata['feat'] = tgt_graph.ndata['feat'].float().to(DEVICE)
            tgt_domain_labels = torch.ones(tgt_graph.batch_size, dtype=torch.long).to(DEVICE)  # 目标领域标记为1

            # 合并输入
            batched_graph = dgl.batch([batch_graph, tgt_graph]).to(DEVICE)
            domain_labels = torch.cat([batch_domain_labels, tgt_domain_labels]).to(DEVICE)
            # 动态掩码生成
            mask = generate_mask(batched_graph, critical_nodes)
            # mask = None

            model.train()

            # _, defect_prob, domain_logits , x_src_mmd, x_tar_mmd, x_loss_mmd_node = model(batched_graph, batch_graph, tgt_graph, alpha, mask=mask, compute_node_mmd=True)
            _, defect_prob, x_src_mmd, x_tar_mmd, x_loss_mmd_node = model(batched_graph, batch_graph, tgt_graph, alpha, mask=mask, compute_node_mmd=True)
            criterion = nn.BCELoss().to(DEVICE)
            batch_y = batch_y.float().view(-1, 1)  # (batch_size, 1)

            # loss_c = criterion(defect_prob, batch_y)
            # 仅计算源领域的缺陷损失
            src_defect_prob = defect_prob[:batch_graph.batch_size]  # 取前batch_size个预测（源领域）

            defect_loss = F.binary_cross_entropy(src_defect_prob, batch_y)

            #MMDLoss
            x_loss_mmd = MMD.mmd_loss(x_src_mmd, x_tar_mmd)

            # 域损失
            # domain_loss = F.cross_entropy(domain_logits, domain_labels)

            # def downsample_tensor(tensor, max_nodes=1000):
            #     """
            #     随机下采样一个 [N, D] 的 tensor，最多保留 max_nodes 行
            #     """
            #     if tensor.size(0) > max_nodes:
            #         idx = torch.randperm(tensor.size(0))[:max_nodes]
            #         return tensor[idx]
            #     else:
            #         return tensor

            # ✅ Attention MMD Loss
            # gate_scores_src = torch.sigmoid(gate_scores_src)
            # gate_scores_tar = torch.sigmoid(gate_scores_tar)

            # gate_scores_src = downsample_tensor(gate_scores_src, max_nodes=10000)
            # gate_scores_tar = downsample_tensor(gate_scores_tar, max_nodes=10000)
            #
            # attention_mmd_loss = MMD.mmd_loss(gate_scores_src, gate_scores_tar)

            # without MMD
            # loss = defect_loss

            # file-level MMD
            loss = defect_loss + x_loss_mmd * 5

            # node-level MMD
            # loss = defect_loss + x_loss_mmd_node * 5

            loss_history['total'].append(loss.item())
            loss_history['defect'].append(defect_loss.item())
            loss_history['mmd'].append(x_loss_mmd.item())

            # print(f'loss: {loss.item():.6f}')
            # print('loss: ' + str(loss))
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()

            total_loss_train += loss.item()
            total_defect_loss += defect_loss.item()
            total_mmd_loss += x_loss_mmd.item()
            # print("label分布: ", domain_labels.sum().item(), "/", domain_labels.size(0))  # 有多少是目标域

            # if step == num_steps_per  _epoch - 1:
            #     domain_diagnostics(domain_logits, domain_labels, step=epoch, show_plot=True)

        avg_defect_loss = total_defect_loss / num_steps_per_epoch
        avg_train_loss = total_loss_train / num_steps_per_epoch
        avg_mmd_loss = total_mmd_loss / num_steps_per_epoch

        epoch_loss_history['total'].append(avg_train_loss)
        epoch_loss_history['defect'].append(avg_defect_loss)
        epoch_loss_history['mmd'].append(avg_mmd_loss)

        # 获取 domain acc
        # domain_preds = domain_logits.argmax(1).detach().cpu()
        # domain_true = domain_labels.detach().cpu()
        # domain_acc = (domain_preds == domain_true).float().mean().item()

        # print(
        #     f"[Epoch {epoch}] Total Loss = {avg_train_loss:.4f} | Defect Loss = {avg_defect_loss:.4f} | Domain Acc = {domain_acc:.4f}")

        print(
            f"[Epoch {epoch}] Total Loss = {avg_train_loss:.4f} | Defect Loss = {avg_defect_loss:.4f}")

        # Early stopping based on defect_loss only
        if avg_defect_loss < best_defect_loss:
            best_defect_loss = avg_defect_loss
            best_model_state = copy.deepcopy(model.state_dict())
            no_improve_epochs = 0
            print(f"✅ Best defect loss updated: {best_defect_loss:.4f}")
        else:
            no_improve_epochs += 1
            print(f"📉 No defect loss improvement: {no_improve_epochs}/{patience}")

        if no_improve_epochs >= patience :
            print(f"🛑 Early stopping at epoch {epoch} (defect_loss converged)")
            break

    # 恢复最优模型
    if best_model_state:
        model.load_state_dict(best_model_state)
        print("✅ Loaded best model based on defect loss.")

        # steps_in_epoch = num_steps_per_epoch
        #
        # avg_total = sum(loss_history['total'][-steps_in_epoch:]) / steps_in_epoch
        # avg_defect = sum(loss_history['defect'][-steps_in_epoch:]) / steps_in_epoch
        # avg_mmd = sum(loss_history['mmd'][-steps_in_epoch:]) / steps_in_epoch
        #
        # epoch_loss_history['total'].append(avg_total)
        # epoch_loss_history['defect'].append(avg_defect)
        # epoch_loss_history['mmd'].append(avg_mmd)

    GNN_acc, GNN_auc, GNN_f1, GNN_mcc, GNN_gmean, GNN_precision, GNN_recall = WGNN_test(critical_nodes, test_tar, model, train_graphs, test_graphs, train_label, test_label, train_hand_craft,
                                                                          test_hand_craft, GNN_acc, GNN_auc, GNN_f1, GNN_mcc, GNN_gmean, GNN_precision, GNN_recall)

    WGNN['acc'].append(GNN_acc)
    WGNN['auc'].append(GNN_auc)
    WGNN['f1'].append(GNN_f1)
    WGNN['mcc'].append(GNN_mcc)
    WGNN['gmean'].append(GNN_gmean)
    WGNN['precision'].append(GNN_precision)
    WGNN['recall'].append(GNN_recall)

avg_acc = np.mean(WGNN['acc'])
avg_auc = np.mean(WGNN['auc'])
avg_f1 = np.mean(WGNN['f1'])
avg_mcc = np.mean(WGNN['mcc'])
avg_gmean = np.mean(WGNN['gmean'])
avg_precision = np.mean(WGNN['precision'])
avg_recall = np.mean(WGNN['recall'])

print(f"Average - F1: {avg_f1:.3f}, Accuracy: {avg_acc:.3f},"
      f" MCC: {avg_mcc:.3f}, AUC: {avg_auc:.3f},"
      f" G-Mean: {avg_gmean:.3f}, Precision: {avg_precision:.3f}, Recall: {avg_recall:.3f}")

# 先构造一行数据（字典形式）
row = {
    "F1": round(avg_f1, 3),
    "Accuracy": round(avg_acc, 3),
    "MCC": round(avg_mcc, 3),
    "AUC": round(avg_auc, 3),
    "G-Mean": round(avg_gmean, 3),
    "Precision": round(avg_precision, 3),
    "Recall": round(avg_recall, 3)
}

# 输出文件路径
csv_path = "fc-average_metrics_log.csv"

# 判断文件是否存在：决定是否写入表头
write_header = not os.path.exists(csv_path)

# 用 pandas 追加到 CSV
pd.DataFrame([row]).to_csv(csv_path, mode='a', index=False, header=write_header)

# plt.figure(figsize=(7, 4))
# plt.plot(epoch_loss_history['total'], label='Total Loss', linewidth=1.5)
# plt.plot(epoch_loss_history['defect'], label='Defect Loss', linewidth=1.5)
# plt.plot(epoch_loss_history['mmd'], label='MMD Loss', linewidth=1.5)
#
# plt.xlabel('Training Epochs (camel-1.6 - xalan-2.7)',fontsize=10)
# plt.ylabel('Loss',fontsize=10)
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('loss_curve_epoch.pdf', format='pdf')
# plt.show()

# End Time
end_time = datetime.datetime.now()
print(end_time - start_time)