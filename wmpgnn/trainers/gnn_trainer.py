from wmpgnn.trainers.trainer import Trainer
from wmpgnn.util.functions import positive_edge_weight, positive_node_weight, weight_four_class, acc_four_class
import torch
from torch import nn
from torch_scatter import scatter_add
import numpy as np
import pandas as pd


def positive_edge_weight(loader):
    sum_edges = 0
    sum_pos = 0
    for data in loader:
        sum_edges += data.edges.shape[0]
        sum_pos  += torch.sum(data.y[:,0]==0).item()
    return sum_edges/(2*sum_pos)

def positive_node_weight(loader):
    sum_nodes = 0
    sum_pos = 0
    for data in loader:
        num_nodes=data.nodes.shape[0]
        #out = data.edges.new_zeros(num_nodes, 4)
        node_sum = scatter_add(data.y,data.senders,dim=0)
        ynodes = (1.*(torch.sum(node_sum[:,1:],1)>0)).unsqueeze(1)
        sum_nodes += num_nodes
        sum_pos  += torch.sum(ynodes==1).item()
    return sum_nodes/(2*sum_pos)


class GNNTrainer(Trainer):
    """ Class for training """

    def __init__(self, config, model, train_loader, val_loader, add_bce=True, use_bce_pos_weight=False):
        super().__init__(config, model, train_loader, val_loader)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        weights = weight_four_class(self.train_loader)
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
        #self.beta_bce_nodes = 3.1
        #self.beta_bce_edges =  33.2256
        self.beta_bce_nodes = 1
        self.beta_bce_edges = 1

        self.ce_train_loss = []
        self.ce_val_loss = []
        self.bce_nodes_train_loss = []
        self.bce_nodes_val_loss = []
        self.bce_edges_train_loss = []
        self.bce_edges_val_loss = []

    def set_beta_bce_nodes(self, beta):
        self.beta_bce_nodes = beta

    def set_beta_bce_edges(self, beta):
        self.beta_BCE_edges = beta

    def eval_one_epoch(self, train=True):
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
            out = data.edges.new_zeros(num_nodes, data.edges.shape[1])
            node_sum = scatter_add(data.y, data.senders, out=out, dim=0)
            ynodes_start = (1. * (torch.sum(node_sum[:, 1:], 1) > 0)).unsqueeze(1)
            label0 = data.y.argmax(dim=1)
            answers = torch.ones_like(data.edges).cuda()

            outputs = self.model(data)
            data = outputs
            label = data.y.argmax(dim=1)
            num_nodes = data.nodes.shape[0]
            out = data.edges.new_zeros(num_nodes, data.edges.shape[1])
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
            acc_one_batch = acc_four_class(outputs.edges, label)
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
        df = pd.DataFrame( data)
        df.to_csv(file_name)
        return df
