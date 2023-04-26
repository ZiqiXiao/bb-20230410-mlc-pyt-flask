import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv, TopKPooling, global_mean_pool, GCNConv
from sklearn.metrics import roc_auc_score
import numpy as np

import importlib.util
import io

def load_dataset(name):
    dataset = TUDataset(root='data/TUDataset', name=name)
    return dataset

def get_stats(dataset):
    num_graphs = len(dataset)
    num_classes = dataset.num_classes
    avg_num_nodes = sum(data.num_nodes for data in dataset) / num_graphs
    num_train_graphs = int(len(dataset) * 0.8)
    num_test_graphs = len(dataset) - num_train_graphs

    nodes = [{'id': i, 'label': str(i)} for i in range(dataset[0].num_nodes)]
    edges = [{'from': edge[0].item(), 'to': edge[1].item(), 'arrows': 'to'} for edge in dataset[0].edge_index.t()]

    return {
        'num_graphs': num_graphs,
        'num_classes': num_classes,
        'avg_num_nodes': avg_num_nodes,
        'num_train_graphs': num_train_graphs,
        'num_test_graphs': num_test_graphs,
        'nodes': nodes,
        'edges': edges
    }

# Define your model here
class GCN(torch.nn.Module):
    def __init__(self, n_ftrs, n_cls, hidden_channels):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(n_ftrs, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, n_cls)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x
    

from torch_geometric.nn import GraphConv

class GraphConvModel(torch.nn.Module):
    def __init__(self, n_ftrs, n_cls, hidden_channels):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = GraphConv(n_ftrs, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, n_cls)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x
    
def train_loop(model, optimizer, criterion, train_loader):
    model.train()
    
    for data in train_loader:  # Iterate in batches over the training dataset.
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
    
    return model

def evl(model, test_loader):
    model.eval()
    
    correct = 0
    probs = []
    true_labels = []

    for data in test_loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.batch)
        pred_probs = torch.softmax(out, dim=1).detach().cpu().numpy() 
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        probs.extend(pred_probs[:, 1])
        true_labels.extend(data.y.cpu().numpy())
    return correct / len(test_loader.dataset), np.array(probs), np.array(true_labels)  # Derive ratio of correct predictions.

def train(dataset, algorithm, socketio, custom_code=None, epochs=100):
    n_ftr = dataset.num_node_features
    n_cls = dataset.num_classes

    if algorithm == 'GCN':
        model = GCN(n_ftr, n_cls, 64)
    elif algorithm == 'GraphConv':
        model = GraphConvModel(n_ftr, n_cls, 64)
    elif algorithm == 'Customized':
        if custom_code:
            custom_locals = {}
            exec(custom_code, globals(), custom_locals)
            model_class = custom_locals.get('CustomModel')
            if model_class is not None:
                model = model_class(n_ftr, n_cls, 64)
            else:
                raise ValueError("CustomModel class not found in the custom_code.")
        else:
            raise ValueError("Customized algorithm selected but custom_code is empty.")
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    # split dataset into train and test
    torch.manual_seed(12345)
    dataset = dataset.shuffle()
    train_dataset = dataset[:int(len(dataset) * 0.8)]
    test_dataset = dataset[int(len(dataset) * 0.8):]
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    for epoch in range(epochs):
        model = train_loop(model, optimizer, criterion, train_loader)
        train_acc, train_probs, train_true_labels = evl(model, train_loader)
        test_acc, test_probs, test_true_labels = evl(model, test_loader)

        train_auc = roc_auc_score(train_true_labels, train_probs, average='macro')
        test_auc = roc_auc_score(test_true_labels, test_probs, average='macro')
        # print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Train AUC: {train_auc:.4f}, Test AUC: {test_auc:.4f}')

        # Emit progress update
        progress = int((epoch + 1) / epochs * 100)
        socketio.emit('training_progress', {'progress': progress, 
                                            'epoch': epoch,
                                            'total_epochs': epochs,
                                            'train_acc': train_acc, 
                                            'test_acc': test_acc,
                                            'train_auc': train_auc,
                                            'test_auc': test_auc})
    
    return model