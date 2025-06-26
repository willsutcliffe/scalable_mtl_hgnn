from wmpgnn.trainers.trainer import Trainer
from wmpgnn.util.functions import hetero_positive_edge_weight, hetero_positive_node_weight, weight_four_class, acc_four_class
import torch
from torch import nn
from torch_scatter import scatter_add
import numpy as np
import pandas as pd

class HeteroGNNTrainer(Trainer):
    """
    Trainer for heterogeneous GNNs with multi-task objectives:
      - Four-class edge classification (LCA task)
      - Binary edge/node auxiliary losses
      - Optional track-to-primary vertex (PV) association

    Inherits from:
        Trainer: abstract base class for training loops.

    Attributes:
        optimizer: Adam optimizer on model parameters.
        criterion: CrossEntropyLoss for LCA task.
        criterion_bce_edges: BCE or BCEWithLogitsLoss for edges.
        criterion_bce_nodes: BCE or BCEWithLogitsLoss for nodes.
        criterion_pvs: BCE or BCEWithLogitsLoss for PV task.
        use_logits: Whether to expect logits for BCE tasks.
        beta_bce_edges, beta_bce_nodes, beta_bce_pvs (float): Scaling factors.
        add_bce, add_pv, no_lca_task (bool): Task flags.
        train_pv_acc, val_pv_acc: Lists of PV-edge accuracy per epoch.
        ce_train_loss, ce_val_loss: CE loss histories.
        bce_edges_train_loss, bce_edges_val_loss: Edge-BCE loss histories.
        bce_nodes_train_loss, bce_nodes_val_loss: Node-BCE loss histories.
        bce_pvs_train_loss, bce_pvs_val_loss: PV-BCE loss histories.
    """
    def __init__(self, config, model, train_loader, val_loader, add_bce=True,
                 use_bce_pos_weight=False, add_pv = True, no_lca_task = False):
        """
        Initialize the HeteroGNNTrainer.

        Args:
            config (dict): Configuration dict (must include 'device').
            model (nn.Module): Heterogeneous GNN model.
            train_loader: DataLoader for training graphs.
            val_loader: DataLoader for validation graphs.
            add_bce (bool): Include BCE losses for edges/nodes.
            use_bce_pos_weight (bool): Use positive-class weighting in BCEWithLogitsLoss.
            add_pv (bool): Include PV association BCE loss.
            no_lca_task (bool): If True, skip the four-class LCA loss.
        """
        super().__init__(config, model, train_loader, val_loader)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        weights = weight_four_class(self.train_loader, hetero=True)
        self.criterion = nn.CrossEntropyLoss(weight=weights)
        pos_weight = hetero_positive_edge_weight(train_loader)
        pos_weight = torch.tensor([pos_weight])

        if use_bce_pos_weight:
            pos_weight = hetero_positive_edge_weight(train_loader)
            pos_weight = torch.tensor([pos_weight])
            self.criterion_bce_edges = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            pos_weight = hetero_positive_node_weight(train_loader)
            pos_weight = torch.tensor([pos_weight])
            self.criterion_bce_nodes = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            # related to average no. of pvs as for each track one pv is correct
            pos_weight = torch.tensor([6.1118])
            self.criterionBCE_pvs = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            self.criterionBCE_pvs.cuda()
            self.use_logits = True
        else:
            self.criterion_bce_edges = nn.BCELoss()
            self.criterion_bce_nodes = nn.BCELoss()
            self.criterion_bce_pvs = nn.BCELoss()
            self.use_logits = False

        print("Use logits ", self.use_logits)
        self.criterion.to('cuda')
        self.criterion_bce_edges.cuda()
        self.criterion_bce_nodes.cuda()
        self.criterion_bce_pvs.cuda()
        self.model.cuda()

        self.add_bce = add_bce
        self.beta_bce_nodes = 3.1
        self.beta_bce_edges =  33.2256
        self.beta_bce_pvs = 1

        self.train_pv_acc = []
        self.val_pv_acc = []

        self.ce_train_loss = []
        self.ce_val_loss = []
        self.bce_nodes_train_loss = []
        self.bce_nodes_val_loss = []
        self.bce_edges_train_loss = []
        self.bce_edges_val_loss = []
        self.bce_pvs_train_loss = []
        self.bce_pvs_val_loss = []
        self.add_pv = add_pv
        self.no_lca_task = no_lca_task

    def set_beta_bce_nodes(self, beta):
        """Set scaling factor for node-level BCE loss."""
        self.beta_BCE_nodes = beta

    def set_beta_bce_edges(self, beta):
        """Set scaling factor for edge-level BCE loss."""
        self.beta_BCE_edges = beta

    def set_beta_bce_pvs(self, beta):
        """Set scaling factor for PV-association BCE loss."""
        self.beta_BCE_pvs= beta

    def eval_one_epoch(self, train=True):
        """
        Evaluate (and train) one epoch over graphs.

        Args:
            train (bool): If True, backprop and update; else eval only.

        Returns:
            last_loss (float): Average total loss from the final batch.
            acc_per_class (Tensor[4]): Mean four-class edge accuracy.
            pv_edge_acc (float): Mean PV-edge binary accuracy.
            pv_node_acc (float): Mean PV-node binary accuracy.
        """
        running_loss = 0.
        last_loss = 0.
        running_ce_loss = 0.
        running_bce_edge_loss = 0.
        running_bce_node_loss = 0.
        running_bce_pv_loss = 0.
        acc_one_epoch = []
        pv_acc_one_epoch = []
        pv_node_acc_one_epoch = []
        if train == True:
            data_loader = self.train_loader
            self.model.train(True)
        else:
            data_loader = self.val_loader
            self.model.train(False)
            self.model.eval()
        last_batch = len(data_loader)

        for i, data in enumerate(data_loader):

            if train:
                self.optimizer.zero_grad()
            data.to('cuda')

            outputs = self.model(data)

            data = outputs

            label = data[('tracks', 'to', 'tracks')].y.argmax(dim=1)
            #pv_label = torch.tensor(data[('tracks', 'to', 'pvs')].y, dtype=torch.float32)
            pv_label = data[('tracks', 'to', 'pvs')].y.to(dtype=torch.float32)
            if not self.no_lca_task:
                loss = self.criterion(outputs[('tracks', 'to', 'tracks')].edges, label)
                running_ce_loss += loss.item()
            else:
                loss = torch.tensor(0.).cuda()
            num_nodes = data['tracks'].x.shape[0]
            out = data[('tracks', 'to', 'tracks')].edges.new_zeros(num_nodes,
                                                                   data[('tracks', 'to', 'tracks')].y.shape[1])
            node_sum = scatter_add(data[('tracks', 'to', 'tracks')].y, data[('tracks', 'to', 'tracks')].edge_index[0],
                                   out=out, dim=0)
            ynodes = (1. * (torch.sum(node_sum[:, 1:], 1) > 0)).unsqueeze(1)
            y_bce = 1. * (data[('tracks', 'to', 'tracks')].y[:, 0] == 0).unsqueeze(1)

            yb = ynodes[data[('tracks', 'to', 'pvs')]['edge_index'][0]] * data[('tracks', 'to', 'pvs')].y
            pv_sum = scatter_add(yb, data[('tracks', 'to', 'pvs')].edge_index[1], dim=0)
            pv_target = 1. * (pv_sum > 0)
            
            for block in self.model._blocks:
                if self.use_logits:
                    if self.add_pv:
                        bce_pvs_loss = (self.beta_bce_pvs  * self.criterion_bce_pvs(block.edge_logits[('tracks', 'to', 'pvs')], pv_label))
                        running_bce_pv_loss += bce_pvs_loss.item()
                        loss +=  bce_pvs_loss
                    if self.add_bce:
                        bce_edges_loss = (self.beta_bce_edges * self.criterion_bce_edges(block.edge_logits[('tracks', 'to', 'tracks')], y_bce))
                        bce_nodes_loss = (self.beta_bce_nodes  * self.criterion_bce_nodes(block.node_logits['tracks'], ynodes))
                        running_bce_edge_loss += bce_edges_loss.item()
                        running_bce_node_loss += bce_nodes_loss.item()
                        loss += bce_edges_loss
                        loss += bce_nodes_loss
                else:
                    if self.add_pv:
                        bce_pvs_loss = (self.beta_bce_pvs * self.criterion_bce_pvs(block.edge_weights[('tracks', 'to', 'pvs')], pv_label))
                        running_bce_pv_loss += bce_pvs_loss.item()
                        loss +=  bce_pvs_loss
                    if self.add_bce:
                        bce_edges_loss = (self.beta_bce_edges * self.criterion_bce_edges(block.edge_weights[('tracks', 'to', 'tracks')], y_bce))
                        bce_nodes_loss = (self.beta_bce_nodes  * self.criterion_bce_nodes(block.node_weights['tracks'], ynodes))
                        running_bce_edge_loss += bce_edges_loss.item()
                        running_bce_node_loss += bce_nodes_loss.item()
                        loss += bce_edges_loss
                        loss += bce_nodes_loss
                    # loss += 1*self.criterionBCEnodes(block.node_logits['pvs'], pv_target)
            acc_one_batch = acc_four_class(outputs[('tracks', 'to', 'tracks')].edges, label)
            acc_one_epoch.append(acc_one_batch)

            pv_acc_one_batch = torch.sum(
                pv_label == (self.model._blocks[-1].edge_weights[('tracks', 'to', 'pvs')] > 0.5)) / pv_label.shape[0]
            pv_acc_one_epoch.append(pv_acc_one_batch)

            pv_node_acc_one_batch = torch.sum(pv_target == (self.model._blocks[-1].node_weights['pvs'] > 0.5)) / \
                                    pv_target.shape[0]
            pv_node_acc_one_epoch.append(pv_node_acc_one_batch)
            if train:
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()
            if (i + 1) == last_batch:
                last_loss = running_loss / last_batch  # loss per batch
                print('  batch {} last_batch {} loss: {}'.format(i + 1, last_batch, last_loss))

                running_loss = 0.

        acc_one_epoch = torch.stack(acc_one_epoch)
        pv_acc_one_epoch = torch.stack(pv_acc_one_epoch)
        pv_node_acc_one_epoch = torch.stack(pv_node_acc_one_epoch)

        if train:
            self.ce_train_loss.append(running_ce_loss/last_batch)
            self.bce_edges_train_loss.append(running_bce_edge_loss/last_batch)
            self.bce_nodes_train_loss.append(running_bce_node_loss/last_batch)
            self.bce_pvs_train_loss.append(running_bce_pv_loss/last_batch)
        else:
            self.bce_edges_val_loss.append(running_bce_edge_loss/last_batch)
            self.bce_nodes_val_loss.append(running_bce_node_loss/last_batch)
            self.ce_val_loss.append(running_ce_loss/last_batch)
            self.bce_pvs_val_loss.append(running_bce_pv_loss/last_batch)
        return last_loss, acc_one_epoch.nanmean(dim=0), pv_acc_one_epoch.nanmean(dim=0).item(), pv_node_acc_one_epoch.nanmean(
            dim=0)

    def train(self, epochs=10, starting_epoch=0, learning_rate=0.001):
        """
        Full training loop over multiple epochs.

        Args:
            epochs (int): Total epochs to run.
            starting_epoch (int): Epoch index to start from (for resuming).
            learning_rate (float): Learning rate for Adam optimizer.
        """
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(starting_epoch, epochs):
            print(f"At epoch {epoch}")
            self.epochs.append(epoch)
            train_loss, train_acc, train_pv_acc, train_B_pv_acc = self.eval_one_epoch()
            self.model.train(False)
            val_loss, val_acc, val_pv_acc, val_B_pv_acc = self.eval_one_epoch(train=False)
            self.train_loss.append(train_loss)
            self.train_acc.append(train_acc)
            self.train_pv_acc.append(train_pv_acc)
            self.val_loss.append(val_loss)
            self.val_acc.append(val_acc)
            self.val_pv_acc.append(val_pv_acc)
            print(f'Train Loss: {train_loss:03f}')
            print(f'Val Loss: {val_loss:03f}')
            print(f'Train Acc: {train_acc}')
            print(f'Val Acc: {val_acc}')
            print(f'Train pv edge Acc: {train_pv_acc}')
            print(f'Val pv edge Acc: {val_pv_acc}')
            print(f'Train pv edge Acc: {train_B_pv_acc}')
            print(f'Val pv edge Acc: {val_B_pv_acc}')

    def save_dataframe(self, file_name):
        """
        Save all recorded metrics to a CSV and return the DataFrame.

        Columns:
          - train_loss, val_loss
          - train_acc_LCA{i}, val_acc_LCA{i} for i in 0..3
          - If add_bce: ce_*, bce_edges_*, bce_nodes_*, bce_pvs_*
          - pv_train_acc, pv_val_acc
        """
        data =  {
            "train_loss":self.train_loss,
            "val_loss":self.val_loss,
            "train_acc_LCA0":list(np.array(self.train_acc)[:,0]),
            "val_acc_LCA0": list(np.array(self.val_acc)[:,0]),
            "train_acc_LCA1":list(np.array(self.train_acc)[:,1]),
            "val_acc_LCA1": list(np.array(self.val_acc)[:,1]),
            "train_acc_LCA2": list(np.array(self.train_acc)[:,2]),
            "val_acc_LCA2": list(np.array(self.val_acc)[:,2]),
            "train_acc_LCA3":list(np.array(self.train_acc)[:,3]),
            "val_acc_LCA3": list(np.array(self.val_acc)[:,3]),
        }
        if self.add_bce:
            data["ce_train_loss"] = self.ce_train_loss
            data["ce_val_loss"]  = self.ce_val_loss
            data["bce_nodes_train_loss"] = self.bce_nodes_train_loss
            data["bce_nodes_val_loss"] = self.bce_nodes_val_loss
            data["bce_edges_train_loss"]  = self.bce_edges_train_loss
            data["bce_edges_val_loss"] = self.bce_edges_val_loss

        data["pv_train_acc"] = self.train_pv_acc
        data["pv_val_acc"] = self.val_pv_acc
        data["pv_train_loss"] = self.bce_pvs_train_loss
        data["pv_val_loss"] = self.bce_pvs_val_loss
        df = pd.DataFrame( data)
        df.to_csv(file_name)
        return df
