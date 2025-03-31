from wmpgnn.trainers.trainer import Trainer
from wmpgnn.util.functions import positive_edge_weight, positive_node_weight, weight_n_class, acc_n_class
import torch
from torch import nn
from torch_scatter import scatter_add
import numpy as np
import pandas as pd


def positive_edge_weight(loader):
    """
    Compute the ratio of total edges to twice the number of positive edges.

    This is useful as a pos_weight for BCE loss to counter class imbalance.

    Args:
        loader (Iterable): A PyTorch DataLoader or other iterator yielding
            graph-batch objects with fields:
              - data.edges: Tensor of shape [E, …]
              - data.y:    Tensor of shape [E, C] where the 0th column
                           indicates the “negative” class (y[:,0] == 0).

    Returns:
        float: (total number of edges) / (2 × number of positive edges).
    """
    sum_edges = 0
    sum_pos = 0
    for data in loader:
        sum_edges += data.edges.shape[0]
        sum_pos  += torch.sum(data.y[:,0]==0).item()
    return sum_edges/(2*sum_pos)

def positive_node_weight(loader):
    """
    Compute the ratio of total nodes to twice the number of “positive” nodes.

    A node is considered “positive” if any of its incident edges (excluding
    the background class at index 0) carry a positive label.

    Args:
        loader (Iterable): An iterator yielding graph-batch objects with:
          - data.nodes:   Tensor of shape [N, …]
          - data.y:       Tensor of shape [E, C]
          - data.senders: LongTensor of shape [E] mapping edges to source nodes.

    Returns:
        float: (total number of nodes) / (2 × number of positive nodes).
    """
    sum_nodes = 0
    sum_pos = 0
    for data in loader:
        num_nodes=data.nodes.shape[0]
        node_sum = scatter_add(data.y,data.senders,dim=0)
        ynodes = (1.*(torch.sum(node_sum[:,1:],1)>0)).unsqueeze(1)
        sum_nodes += num_nodes
        sum_pos  += torch.sum(ynodes==1).item()
    return sum_nodes/(2*sum_pos)


class GNNTrainer(Trainer):
    """
    Trainer for graph-structured data leveraging both Cross-Entropy and
    optional BCE losses on edges and nodes.

    Inherits:
        Trainer (abstract base class for epoch/train loop management).

    Key Features:
      - Four-class cross-entropy on edges.
      - Optional BCE/BCEWithLogits weighting for edge/node imbalance.
      - Tracks separate losses (CE, BCE-edges, BCE-nodes) for train & val.
      - Per-epoch accuracy logging for four classes.

    Args:
        config (dict): Hyperparameter dict, must include:
            - 'device':     'cuda' or 'cpu'
            - (other keys as defined by base Trainer)
        model (torch.nn.Module): GNN implementing a forward(data) → data
            with updated `data.edges` logits/weights and `data.y`.
        train_loader, val_loader (Iterable): DataLoaders yielding batched
            graph objects with attributes:
              - data.edges, data.nodes, data.y (edge labels), senders, receivers.
        add_bce (bool, default=True): Include BCE edge/node losses.
        use_bce_pos_weight (bool, default=False):
            If True, use `positive_edge_weight` / `positive_node_weight`
            to set `pos_weight` in `BCEWithLogitsLoss`; otherwise plain BCE.

    Attributes:
        optimizer:                Adam optimizer on `model.parameters()`.
        criterion:                CrossEntropyLoss with class weights.
        criterion_bce_edges:      BCE(BCEWithLogits) on edges.
        criterion_bce_nodes:      BCE(BCEWithLogits) on nodes.
        beta_bce_edges, beta_bce_nodes (float):
                                  Scaling factors for BCE losses.
        use_logits (bool):        True if using logits‐based BCEWithLogits.
        add_bce (bool):           Whether to include BCE terms.
        ce_train_loss, ce_val_loss,
        bce_edges_train_loss, bce_edges_val_loss,
        bce_nodes_train_loss, bce_nodes_val_loss (lists):
                                  Loss histories.
    """
    def __init__(self, config, model, train_loader, val_loader, add_bce=True, use_bce_pos_weight=False):
        """
        Initialize a GNNTrainer for graph‐structured learning with both
        cross‐entropy and optional BCE losses.

        Args:
            config (dict):
                A configuration dictionary. Must include at least:
                  - 'device': str, either 'cuda' or 'cpu'
                  - (other keys consumed by your model/trainer)
            model (torch.nn.Module):
                The GNN model instance. Must accept a BatchedData input
                and return an output with updated `.edges`, `.nodes`, and `.y`.
            train_loader (DataLoader):
                Yields training graph batches with fields:
                  - `.edges`, `.nodes`, `.y`, `.senders`, `.receivers`etc
            val_loader (DataLoader):
                Yields validation graph batches (same structure as train_loader).
            add_bce (bool, default=True):
                If True, include BCE (or BCEWithLogits) losses for
                per-edge and per-node auxiliary supervision.
            use_bce_pos_weight (bool, default=False):
                If True, compute `pos_weight` from training data imbalance
                (via `positive_edge_weight` / `positive_node_weight`) and
                use `BCEWithLogitsLoss`. Otherwise use unweighted `BCELoss`.

        """
        super().__init__(config, model, train_loader, val_loader)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        weights = weight_n_class(self.train_loader, n_class=self.LCA_classes)
        self.criterion = nn.CrossEntropyLoss(weight=weights)

        if use_bce_pos_weight:
            pos_weight = positive_edge_weight(train_loader)
            pos_weight = torch.tensor([pos_weight])
            self.criterion_bce_edges = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            pos_weight = positive_node_weight(train_loader)
            pos_weight = torch.tensor([pos_weight])
            self.criterion_bce_nodes = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            self.use_logits = True
        else:
            self.criterion_bce_edges = nn.BCELoss()
            self.criterion_bce_nodes = nn.BCELoss()
            self.use_logits = False

        self.criterion.to('cuda')
        self.criterion_bce_edges.cuda()
        self.criterion_bce_nodes.cuda()
        self.model.cuda()

        self.add_bce = add_bce
        self.beta_bce_nodes = 3.1
        self.beta_bce_edges =  33.2256


        
        self.ce_train_loss = []
        self.ce_val_loss = []
        self.bce_nodes_train_loss = []
        self.bce_nodes_val_loss = []
        self.bce_edges_train_loss = []
        self.bce_edges_val_loss = []

    def set_beta_bce_nodes(self, beta):
        """Adjust weight of node-level BCE loss."""
        self.beta_bce_nodes = beta

    def set_beta_bce_edges(self, beta):
        """Adjust weight of edge-level BCE loss."""
        self.beta_BCE_edges = beta

    def eval_one_epoch(self, train=True):
        """
        Run one epoch of forward (and backward if training).

        - Computes CE loss on per-edge four-class logits.
        - Optionally computes BCE/BCEWithLogits on per-edge weights and
          per-node weights from each GNN block.
        - Aggregates per-batch four-class accuracy via `acc_four_class`.

        Returns:
            last_loss (float): Average total loss of the last batch.
            class_acc (Tensor[4]): Mean accuracy for each of four classes.
        """
        running_loss = 0.
        running_ce_loss = 0.
        running_bce_edge_loss = 0.
        running_bce_node_loss = 0.
        last_loss = 0.
        acc_one_epoch = []
        if train == True:
            data_loader = self.train_loader
        else:
            data_loader = self.val_loader
        last_batch = len(data_loader)
        for i, data in enumerate(data_loader):
            data['graph_globals'] = data['graph_globals'].unsqueeze(1)
            data.receivers = data.receivers - torch.min(data.receivers)
            data.senders = data.senders - torch.min(data.senders)
            data.edgepos = data.edgepos - torch.min(data.edgepos)
            if train:
                self.optimizer.zero_grad()

            data.to('cuda')
            yBCE_start = 1. * (data.y[:, 0] == 0).unsqueeze(1)
            num_nodes = data.nodes.shape[0]
            out = data.edges.new_zeros(num_nodes, data.y.shape[1])
            node_sum = scatter_add(data.y, data.senders, out=out, dim=0)
            ynodes_start = (1. * (torch.sum(node_sum[:, 1:], 1) > 0)).unsqueeze(1)
            label0 = data.y.argmax(dim=1)
            answers = torch.ones_like(data.edges).cuda()

            outputs = self.model(data)
            data = outputs
            label = data.y.argmax(dim=1)
            num_nodes = data.nodes.shape[0]
            out = data.edges.new_zeros(num_nodes, data.y.shape[1])
            
            node_sum = scatter_add(data.y, data.senders, out=out, dim=0)
            ynodes = (1. * (torch.sum(node_sum[:, 1:], 1) > 0)).unsqueeze(1)
            loss = self.criterion(outputs.edges, label)
            running_ce_loss += loss.item()
            y_bce= 1. * (data.y[:, 0] == 0).unsqueeze(1)
            if self.add_bce:
                for block in self.model._blocks:
                    if self.use_logits:
                        bce_edge_loss = self.beta_bce_edges * self.criterion_bce_edges(block._network.edge_logits, y_bce)
                        bce_node_loss = self.beta_bce_nodes * self.criterion_bce_nodes(block._network.node_logits, ynodes)
                    else:
                        bce_edge_loss = self.beta_bce_edges * self.criterion_bce_edges(block._network.edge_weights, y_bce)
                        bce_node_loss = self.beta_bce_nodes * self.criterion_bce_nodes(block._network.node_weights, ynodes)
                    running_bce_edge_loss += bce_edge_loss.item()
                    running_bce_node_loss += bce_node_loss.item()
                    loss += bce_edge_loss
                    loss += bce_node_loss
            acc_one_batch = acc_n_class(outputs.edges, label, n_class=data.y.shape[1])
            acc_one_epoch.append(acc_one_batch)
            if train:
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()
            if (i + 1) == last_batch:
                last_loss = running_loss / last_batch  # loss per batch
                print('  batch {} last_batch {} loss: {}'.format(i + 1, last_batch, last_loss))

                running_loss = 0.

        acc_one_epoch = torch.stack(acc_one_epoch)
        if train:
            self.ce_train_loss.append(running_ce_loss/last_batch)
            self.bce_edges_train_loss.append(running_bce_edge_loss/last_batch)
            self.bce_nodes_train_loss.append(running_bce_node_loss/last_batch)
        else:
            self.bce_edges_val_loss.append(running_bce_edge_loss/last_batch)
            self.bce_nodes_val_loss.append(running_bce_node_loss/last_batch)
            self.ce_val_loss.append(running_ce_loss/last_batch)


        return last_loss, acc_one_epoch.nanmean(dim=0)

    def train(self, epochs=10, starting_epoch=0, learning_rate=0.001):
        """
        Full training loop.

        Args:
            epochs (int): Total number of epochs to run (upper bound).
            starting_epoch (int): Initial epoch index (for resume).
            learning_rate (float): Learning rate for Adam optimizer.
        """
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(starting_epoch, epochs):
            print(f"At epoch {epoch}")
            self.epochs.append(epoch)
            train_loss, train_acc = self.eval_one_epoch()
            self.model.train(False)
            val_loss, val_acc = self.eval_one_epoch(train=False)
            self.train_loss.append(train_loss)
            self.train_acc.append(train_acc)
            self.val_loss.append(val_loss)
            self.val_acc.append(val_acc)
            print(f'Train Loss: {train_loss:03f}')
            print(f'Val Loss: {val_loss:03f}')
            print(f'Train Acc: {train_acc}')
            print(f'Val Acc: {val_acc}')

    def save_dataframe(self, file_name):
        """
        Export all tracked metrics to CSV and return the DataFrame.

        Columns include:
          - train_loss, val_loss
          - train_acc_LCA{i}, val_acc_LCA{i} for i in 0..3
          - (if add_bce) ce_train_loss, ce_val_loss,
                        bce_edges_train_loss, bce_edges_val_loss,
                        bce_nodes_train_loss, bce_nodes_val_loss

        Returns:
            pandas.DataFrame: Containing all history columns.
        """
        data =  {
            "train_loss":self.train_loss,
            "val_loss":self.val_loss,
        }
        for nLCA in range(self.LCA_classes):
            data[f"train_acc_LCA{nLCA}"] = list(np.array(self.train_acc)[:, nLCA])
            data[f"val_acc_LCA{nLCA}"] = list(np.array(self.val_acc)[:, nLCA])
            
        if self.add_bce:
            data["ce_train_loss"] = self.ce_train_loss
            data["ce_val_loss"]  = self.ce_val_loss
            data["bce_nodes_train_loss"] = self.bce_nodes_train_loss
            data["bce_nodes_val_loss"] = self.bce_nodes_val_loss
            data["bce_edges_train_loss"]  = self.bce_edges_train_loss
            data["bce_edges_val_loss"] = self.bce_edges_val_loss
        df = pd.DataFrame( data)
        df.to_csv(file_name)
        return df
