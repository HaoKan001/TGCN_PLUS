import math
from sklearn.metrics import matthews_corrcoef, roc_auc_score, f1_score, accuracy_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from imblearn.metrics import geometric_mean_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from torch import Tensor
import numpy as np
import torch
import dgl
import TCA
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from sklearn.decomposition import PCA
from select_features import *
from TriStage_Stage3 import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_mask(g, critical_nodes):
    """生成动态掩码：节点类型在 critical_nodes 中的位置为1，否则为0"""
    node_types = g.ndata['feat']  # 假设节点类型字段是 'type'
    critical_nodes_tensor = torch.tensor(critical_nodes, device=node_types.device)
    mask = torch.isin(node_types, critical_nodes_tensor).float()
    # mask = torch.isin(node_types, torch.tensor(critical_nodes)).float()  # [num_nodes]
    return mask.to(DEVICE)

def WGNN_test(critical_nodes, test_tar, model, train_graph, test_graph, train_label, test_label, train_hand_craft, test_hand_craft, acc, auc, f1, mcc, gmean, precision, recall):

    alpha = 1
    train_graph_mmd = dgl.batch(train_graph)  # 合并训练图
    test_graph_mmd = dgl.batch(test_graph)  # 合并测试图

    Train_mask = generate_mask(train_graph_mmd, critical_nodes)
    Test_mask = generate_mask(test_graph_mmd, critical_nodes)

    # Train_mask = None
    # Test_mask = None
    model.eval()
    with torch.no_grad():
        # train_x, _, _ ,_ , _, _, = model(train_graph_mmd, train_graph_mmd, test_graph_mmd, alpha, mask=Train_mask, compute_node_mmd=False)
        # test_x, _, _ ,_ , _, _, = model(test_graph_mmd, train_graph_mmd, test_graph_mmd, alpha, mask=Test_mask, compute_node_mmd=False)
        train_x, _, _ ,_ , _, = model(train_graph_mmd, train_graph_mmd, test_graph_mmd, alpha, mask=Train_mask, compute_node_mmd=False)
        test_x, _, _ ,_ , _, = model(test_graph_mmd, train_graph_mmd, test_graph_mmd, alpha, mask=Test_mask, compute_node_mmd=False)

    if type(train_x) is Tensor:
        train_x = train_x.data.cpu().numpy()
        test_x = test_x.data.cpu().numpy()

    if type(train_label) is Tensor:
        train_label = train_label.data.cpu().numpy()
        test_label = test_label.data.cpu().numpy()

    if type(train_hand_craft) is Tensor:
        train_hand_craft = train_hand_craft.data.cpu().numpy()
        test_hand_craft = test_hand_craft.data.cpu().numpy()

    train_x = np.concatenate((train_x, train_hand_craft), axis=1)
    test_x = np.concatenate((test_x, test_hand_craft), axis=1)

    # Z-score
    train_x = (train_x - np.mean(train_x, axis=0)) / np.std(train_x, axis=0)
    test_x = (test_x - np.mean(test_x, axis=0)) / np.std(test_x, axis=0)

    idx, train_x, test_x, info = select_features_source_only( train_x, train_label.ravel(), test_x, top_k=64, lambda_shift=2, scale=True )

    train_x = coral_align(train_x, test_x)

    # print("选中特征数:", len(idx))
    # print("前10个特征索引:", idx[:10])

    # 合并特征和标签
    # X = np.concatenate([train_x, test_x], axis=0)
    # domain_labels = np.array([0] * train_x.shape[0] + [1] * test_x.shape[0])  # 0:源域，1:目标域
    #
    # # 运行 t-SNE
    # tsne = TSNE(n_components=2, random_state=42)
    # X_2d = tsne.fit_transform(X)
    #
    # # 绘图
    # plt.figure(figsize=(8, 6))
    # for domain, color, label_name in zip([0, 1], ['blue', 'orange'], ['Source features', 'Target features']):
    #     idx = (domain_labels == domain)
    #     plt.scatter(X_2d[idx, 0], X_2d[idx, 1], c=color, label=label_name, alpha=0.6, s=20)
    #
    # plt.legend()
    # plt.title("t-SNE visualization of source and target features without MMD alignment")
    # # plt.title("t-SNE visualization of source and target features after MMD and GRL")
    # plt.grid(True)
    # plt.savefig("ESE-Source vs Target features without MMD alignment.pdf", format="pdf")
    # plt.show()

    # PCA降维
    pca = PCA(n_components= 0.2)
    pca.fit(train_x)
    train_x_pca = pca.transform(train_x)
    test_x_pca = pca.transform(test_x)

    # train_x_pca = coral_align2(train_x_pca, test_x_pca)

    # print("Shape:", train_x_pca.shape)  # 看降维后的维度
    # print("前10条数据：")
    # print(np.round(train_x_pca[:10], 4))  # 保留4位小数方便看
    #
    # print("n_components_:", pca.n_components_)

    # pca = PCA(n_components=0.3)
    # pca.fit(np.vstack([train_x, test_x]))
    # train_x_pca = pca.transform(train_x)
    # test_x_pca = pca.transform(test_x)

    # train_x_pca = np.concatenate((train_x_pca, train_hand_craft), axis=1)
    # test_x_pca = np.concatenate((test_x_pca, test_hand_craft), axis=1)

    # 3. 使用 t-SNE 可视化降维前后的数据
    # 原始数据的 t-SNE
    # tsne_raw = TSNE(n_components=2, random_state=42)
    # # 注意：t-SNE 应该在训练集和测试集上一起拟合
    # combined_x = np.concatenate([train_x, test_x], axis=0)
    # combined_tsne_raw = tsne_raw.fit_transform(combined_x)
    #
    # # 降维后的 PCA 数据的 t-SNE
    # tsne_pca = TSNE(n_components=2, random_state=42)
    # combined_x_pca = np.concatenate([train_x_pca, test_x_pca], axis=0)
    # combined_tsne_pca = tsne_pca.fit_transform(combined_x_pca)
    #
    # # 4. 绘制图形：原始数据 vs PCA 降维数据
    # plt.figure(figsize=(12, 6))
    #
    # # 原始数据的 t-SNE 可视化
    # plt.subplot(1, 2, 1)
    # plt.scatter(combined_tsne_raw[:train_x.shape[0], 0], combined_tsne_raw[:train_x.shape[0], 1], c='blue',
    #             label='Source features', alpha=0.5)
    # plt.scatter(combined_tsne_raw[train_x.shape[0]:, 0], combined_tsne_raw[train_x.shape[0]:, 1], c='orange',
    #             label='Target features', alpha=0.5)
    # plt.title("t-SNE on Original Features")
    # plt.legend()
    #
    # # PCA 降维后的 t-SNE 可视化
    # plt.subplot(1, 2, 2)
    # plt.scatter(combined_tsne_pca[:train_x_pca.shape[0], 0], combined_tsne_pca[:train_x_pca.shape[0], 1], c='blue',
    #             label='Source features', alpha=0.5)
    # plt.scatter(combined_tsne_pca[train_x_pca.shape[0]:, 0], combined_tsne_pca[train_x_pca.shape[0]:, 1], c='orange',
    #             label='Target features', alpha=0.5)
    # plt.title("t-SNE on PCA Reduced Features")
    # plt.legend()
    # plt.tight_layout()
    #
    # # 保存图形为 PDF 格式
    # plt.savefig("ESE-tsne_visualization.pdf", format="pdf")
    # plt.show()

    # Perform Prediction
    cls = LogisticRegression(max_iter=1000, class_weight="balanced")
    # cls = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
    # cls = SVC(probability=True, kernel='rbf', C=1.0, gamma='scale', random_state=42)
    # cls = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, activation='relu', solver='adam', random_state=42)
    # cls = DecisionTreeClassifier(max_depth=None, random_state=42)
    # cls = KNeighborsClassifier(n_neighbors=5, metric='euclidean', weights='distance')
    # cls = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    # cls = QuadraticDiscriminantAnalysis()

    cls.fit(train_x_pca, train_label.ravel())  # 使用 ravel() 调整标签的形状
    y_pred = cls.predict(test_x_pca)
    # 获取概率预测
    y_proba = cls.predict_proba(test_x_pca)[:, 1]

    # noPCA
    # cls.fit(train_x, train_label.ravel())  # 使用 ravel() 调整标签的形状
    # y_pred = cls.predict(test_x)
    # # 获取概率预测
    # y_proba = cls.predict_proba(test_x)[:, 1]

    # y_pred, Zs, Zt, Ps, Pt = unsupervised_stage3(train_x_pca, train_label.ravel(), test_x_pca, r=20, knn_t=8, knn_cross=8, n_iter=4)

    # Save Result
    acc.append(accuracy_score(y_true=test_label.ravel(), y_pred=y_pred))
    auc.append(roc_auc_score(y_true=test_label.ravel(), y_score=y_proba))
    f1.append(f1_score(y_true=test_label.ravel(), y_pred=y_pred))
    mcc.append(matthews_corrcoef(y_true=test_label.ravel(), y_pred=y_pred))
    gmean.append(geometric_mean_score(y_true=test_label.ravel(), y_pred=y_pred))
    precision.append(precision_score(y_true=test_label.ravel(), y_pred=y_pred))
    recall.append(recall_score(y_true=test_label.ravel(), y_pred=y_pred))
    # tn, fp, fn, tp = confusion_matrix(y_true = test_label.ravel(), y_pred=y_pred, labels=[0, 1]).ravel()
    # # 保留两位小数打印
    # print(f"TP: {tp:.2f}")
    # print(f"FP: {fp:.2f}")
    # print(f"FN: {fn:.2f}")
    # print(f"TN: {tn:.2f}")
    return acc, auc, f1, mcc, gmean, precision, recall


    # 合并特征和标签
    # X = np.concatenate([train_x, test_x], axis=0)
    # domain_labels = np.array([0] * train_x.shape[0] + [1] * test_x.shape[0])  # 0:源域，1:目标域
    #
    # # 运行 t-SNE
    # tsne = TSNE(n_components=2, random_state=42)
    # X_2d = tsne.fit_transform(X)
    #
    # # 绘图
    # plt.figure(figsize=(8, 6))
    # for domain, color, label_name in zip([0, 1], ['blue', 'orange'], ['Source', 'Target']):
    #     idx = (domain_labels == domain)
    #     plt.scatter(X_2d[idx, 0], X_2d[idx, 1], c=color, label=label_name, alpha=0.6, s=20)
    #
    # plt.legend()
    # plt.title("t-SNE visualization of Source vs Target features")
    # plt.xlabel("t-SNE dim 1")
    # plt.ylabel("t-SNE dim 2")
    # plt.grid(True)
    # plt.show()


    # # 3. 使用 t-SNE 可视化降维前后的数据
    # # 原始数据的 t-SNE
    # tsne_raw = TSNE(n_components=2, random_state=42)
    # # 注意：t-SNE 应该在训练集和测试集上一起拟合
    # combined_x = np.concatenate([train_x, test_x], axis=0)
    # combined_tsne_raw = tsne_raw.fit_transform(combined_x)
    #
    # # 降维后的 PCA 数据的 t-SNE
    # tsne_pca = TSNE(n_components=2, random_state=42)
    # combined_x_pca = np.concatenate([train_x_pca, test_x_pca], axis=0)
    # combined_tsne_pca = tsne_pca.fit_transform(combined_x_pca)
    #
    # # 4. 绘制图形：原始数据 vs PCA 降维数据
    # plt.figure(figsize=(12, 6))
    #
    # # 原始数据的 t-SNE 可视化
    # plt.subplot(1, 2, 1)
    # plt.scatter(combined_tsne_raw[:train_x.shape[0], 0], combined_tsne_raw[:train_x.shape[0], 1], c='blue',
    #             label='Source features', alpha=0.5)
    # plt.scatter(combined_tsne_raw[train_x.shape[0]:, 0], combined_tsne_raw[train_x.shape[0]:, 1], c='orange',
    #             label='Target features', alpha=0.5)
    # plt.title("t-SNE on Original Features")
    # plt.legend()
    #
    # # PCA 降维后的 t-SNE 可视化
    # plt.subplot(1, 2, 2)
    # plt.scatter(combined_tsne_pca[:train_x_pca.shape[0], 0], combined_tsne_pca[:train_x_pca.shape[0], 1], c='blue',
    #             label='Source features', alpha=0.5)
    # plt.scatter(combined_tsne_pca[train_x_pca.shape[0]:, 0], combined_tsne_pca[train_x_pca.shape[0]:, 1], c='orange',
    #             label='Target features', alpha=0.5)
    # plt.title("t-SNE on PCA Reduced Features")
    # plt.legend()
    # plt.tight_layout()
    #
    # # 保存图形为 PDF 格式
    # plt.savefig("ESE-tsne_visualization.pdf", format="pdf")
    # plt.show()