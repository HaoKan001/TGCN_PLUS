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
    node_types = g.ndata['feat']
    critical_nodes_tensor = torch.tensor(critical_nodes, device=node_types.device)
    mask = torch.isin(node_types, critical_nodes_tensor).float()
    return mask.to(DEVICE)

def WGNN_test(critical_nodes, test_tar, model, train_graph, test_graph, train_label, test_label, train_hand_craft, test_hand_craft, acc, auc, f1, mcc, gmean, precision, recall):

    alpha = 1
    train_graph_mmd = dgl.batch(train_graph)
    test_graph_mmd = dgl.batch(test_graph)

    Train_mask = generate_mask(train_graph_mmd, critical_nodes)
    Test_mask = generate_mask(test_graph_mmd, critical_nodes)
    model.eval()
    with torch.no_grad():
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

    train_x = coral_align(train_x, test_x)

    pca = PCA(n_components= 0.2)
    pca.fit(train_x)
    train_x_pca = pca.transform(train_x)
    test_x_pca = pca.transform(test_x)

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
    y_proba = cls.predict_proba(test_x_pca)[:, 1]

    # noPCA
    # cls.fit(train_x, train_label.ravel())
    # y_pred = cls.predict(test_x)
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
    return acc, auc, f1, mcc, gmean, precision, recall
