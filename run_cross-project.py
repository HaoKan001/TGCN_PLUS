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
    node_types = g.ndata['feat']
    critical_nodes_tensor = torch.tensor(critical_nodes, device=node_types.device)
    mask = torch.isin(node_types, critical_nodes_tensor).float()
    return mask.to(DEVICE)
    
# Initial super-parameter of network
init_gnn_params = {'EMBED_DIM': 32, 'N_HIDDEN_NODE': 100, 'N_EPOCH': 10, 'LEARNING_RATE': 5e-5,
                   'MOMEMTUN': 0.9, 'L2_WEIGHT': 0.005, 'DROPOUT': 0.5, 'STRIDE': 1, 'PADDING': 0, 'POOL_SIZE': 2,'DICT_SIZE':0, 'TOKEN_SIZE': 0}

# Adjustable parameter
REGENERATE = False
dump_data_path = '/root/autodl-tmp/GAT/'

# Fixed parameter
IMBALANCE_PROCESSOR = RandomOverSampler()  # RandomOverSampler(), RandomUnderSampler(), None, 'cost'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    if os.path.exists(path_train_and_test_set) and not REGENERATE:
        obj = load_data(path_train_and_test_set)
        [train_graphs, test_graphs, train_ast, train_hand_craft, train_label, test_ast, test_hand_craft, test_label, vector_len, vocabulary_size] = obj
    else:
        # Get a list of instances of the training and test sets
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

        # Create graphs from AST with self-loops
        def create_graphs_from_ast(project_root_path, file_instances, package_heads, device, vocabulary=None):
            dict_graphs = extract_data_and_generate_graphs(project_root_path, file_instances, package_heads, graph_vocabulary = vocabulary)

            graphs = []
            missing_files = []

            for qualified_name, graph in dict_graphs:
                if graph is not None:
                    graph = graph.to(device)
                    graphs.append(graph)
                else:
                    missing_files.append(qualified_name)

            if missing_files:
                print("Files without graphs:", missing_files[:10])

            return graphs

        # Create graphs for training and testing data
        train_graphs = create_graphs_from_ast(path_train_source, balanced_file_instances, package_heads, DEVICE, vocabulary = vocabulary)

        test_graphs = create_graphs_from_ast(path_test_source, dict_token_test.keys(), package_heads, DEVICE, vocabulary = vocabulary)

        # Saved to dump_data
        obj = [train_graphs, test_graphs, train_ast, train_hand_craft, train_label, test_ast, test_hand_craft, test_label, vector_len, vocabulary_size]
        dump_data(path_train_and_test_set, obj)

    # ZScore
    train_hand_craft = (train_hand_craft - np.mean(train_hand_craft, axis=0)) / np.std(train_hand_craft, axis=0)
    test_hand_craft = (test_hand_craft - np.mean(test_hand_craft, axis=0)) / np.std(test_hand_craft, axis=0)

    train_src = load_data(os.path.join('/root/autodl-tmp/', f'{train_project_name}_indices.pkl'))
    critical_nodes = train_src

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

    loss_history = {
        'total': [],
        'defect': [],
        'mmd': []
    }

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

    optimizer = optim.Adam(model.parameters(), lr=gnn_params['LEARNING_RATE'], betas=(0.9, 0.999), eps=1e-8, weight_decay=gnn_params['L2_WEIGHT'], amsgrad=False)

    for epoch in range(gnn_params['N_EPOCH']):
        total_defect_loss = 0
        total_loss_train = 0
        total_mmd_loss = 0
        alpha = 0.5 * (1 + torch.cos(torch.tensor(epoch / gnn_params['N_EPOCH'] * 3.1416)))

        num_steps_per_epoch = len(train_loader)

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

            tgt_graph = tgt_graph.to(DEVICE)
            tgt_graph.ndata['feat'] = tgt_graph.ndata['feat'].float().to(DEVICE)
            tgt_domain_labels = torch.ones(tgt_graph.batch_size, dtype=torch.long).to(DEVICE)

            batched_graph = dgl.batch([batch_graph, tgt_graph]).to(DEVICE)
            domain_labels = torch.cat([batch_domain_labels, tgt_domain_labels]).to(DEVICE)
            mask = generate_mask(batched_graph, critical_nodes)
            model.train()
            _, defect_prob, x_src_mmd, x_tar_mmd, x_loss_mmd_node = model(batched_graph, batch_graph, tgt_graph, alpha, mask=mask)
            criterion = nn.BCELoss().to(DEVICE)
            batch_y = batch_y.float().view(-1, 1)  # (batch_size, 1)

            src_defect_prob = defect_prob[:batch_graph.batch_size]

            defect_loss = F.binary_cross_entropy(src_defect_prob, batch_y)

            #MMDLoss
            x_loss_mmd = MMD.mmd_loss(x_src_mmd, x_tar_mmd)

            # file-level MMD
            loss = defect_loss + x_loss_mmd * 5

            loss_history['total'].append(loss.item())
            loss_history['defect'].append(defect_loss.item())
            loss_history['mmd'].append(x_loss_mmd.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_train += loss.item()
            total_defect_loss += defect_loss.item()
            total_mmd_loss += x_loss_mmd.item()
            
        avg_defect_loss = total_defect_loss / num_steps_per_epoch
        avg_train_loss = total_loss_train / num_steps_per_epoch
        avg_mmd_loss = total_mmd_loss / num_steps_per_epoch

        epoch_loss_history['total'].append(avg_train_loss)
        epoch_loss_history['defect'].append(avg_defect_loss)
        epoch_loss_history['mmd'].append(avg_mmd_loss)
        
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
            
    if best_model_state:
        model.load_state_dict(best_model_state)
        print("✅ Loaded best model based on defect loss.")


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

row = {
    "F1": round(avg_f1, 3),
    "Accuracy": round(avg_acc, 3),
    "MCC": round(avg_mcc, 3),
    "AUC": round(avg_auc, 3),
    "G-Mean": round(avg_gmean, 3),
    "Precision": round(avg_precision, 3),
    "Recall": round(avg_recall, 3)
}

csv_path = "average_metrics_log.csv"

write_header = not os.path.exists(csv_path)

pd.DataFrame([row]).to_csv(csv_path, mode='a', index=False, header=write_header)

# End Time
end_time = datetime.datetime.now()

print(end_time - start_time)
