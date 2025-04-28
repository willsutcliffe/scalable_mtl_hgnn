from wmpgnn.trainers.trainer import Trainer
from wmpgnn.util.functions import positive_edge_weight, positive_node_weight, weight_n_class, acc_n_class, eff_n_class, rej_n_class
import torch
from torch import nn
from torch_scatter import scatter_add
import numpy as np
import pandas as pd

from datetime import datetime

def NOW(fmt="%H:%M:%S"):
    """return current time formatted"""
    return datetime.now().strftime(fmt)

def msg(obj):
    """print string with time information"""
    print("[{}] ".format(NOW()),obj)
    
    
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

    def save_checkpoint(self,file_path:str):
        """Saves the model and optimizer state to a checkpoint file."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'criterion_state_dict': self.criterion.state_dict(),
            'criterion_bce_nodes_state_dict': self.criterion_bce_nodes.state_dict(),
            'criterion_bce_edges_state_dict': self.criterion_bce_edges.state_dict(),
            'epoch_warmstart': self.epoch_warmstart,
            'history': self.get_history(),
        }
        torch.save(checkpoint, file_path)
        print(f"Checkpoint saved to {file_path}")
    
    def load_checkpoint(self, file_path=None):
        """Loads the model and optimizer state from a checkpoint file."""
        checkpoint = torch.load(file_path, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.criterion.load_state_dict(checkpoint['criterion_state_dict'])
        self.criterion_bce_nodes.load_state_dict(checkpoint['criterion_bce_nodes_state_dict'])
        self.criterion_bce_edges.load_state_dict(checkpoint['criterion_bce_edges_state_dict'])
        self.set_history(checkpoint['history'])
        self.epoch_warmstart = checkpoint['epoch_warmstart']+1

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
        acc_one_epoch, acc_one_epoch_err = [],[]
        eff_one_epoch, eff_one_epoch_err = [],[]
        rej_one_epoch, rej_one_epoch_err = [],[]
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
            #acc_one_epoch_err.append(acc_err)
            eff_one_batch = eff_n_class(outputs.edges, label, n_class=data.y.shape[1])
            eff_one_epoch.append(eff_one_batch)
            #eff_one_epoch_err.append(eff_err)
            rej_one_batch = rej_n_class(outputs.edges, label, n_class=data.y.shape[1])
            rej_one_epoch.append(rej_one_batch)
            #rej_one_epoch_err.append(rej_err)
            if train:
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()
            if (i + 1) == last_batch:
                last_loss = running_loss / last_batch  # loss per batch
                info_msg = '  batch {} last_batch {} loss: {}'.format(i + 1, last_batch, last_loss)
                msg(info_msg)
                running_loss = 0.

        acc_one_epoch = torch.stack(acc_one_epoch)
        #acc_one_epoch_err = torch.stack(acc_one_epoch_err)
        eff_one_epoch = torch.stack(eff_one_epoch)
        #eff_one_epoch_err = torch.stack(eff_one_epoch_err)
        rej_one_epoch = torch.stack(rej_one_epoch)
        #rej_one_epoch_err = torch.stack(rej_one_epoch_err)
        if train:
            self.ce_train_loss.append(running_ce_loss/last_batch)
            self.bce_edges_train_loss.append(running_bce_edge_loss/last_batch)
            self.bce_nodes_train_loss.append(running_bce_node_loss/last_batch)
        else:
            self.bce_edges_val_loss.append(running_bce_edge_loss/last_batch)
            self.bce_nodes_val_loss.append(running_bce_node_loss/last_batch)
            self.ce_val_loss.append(running_ce_loss/last_batch)

        metrics = {
            'loss': last_loss,
            'acc': acc_one_epoch.nanmean(dim=0),
            'acc_err': acc_one_epoch.std(dim=0),
            'eff': eff_one_epoch.nanmean(dim=0),
            'eff_err': eff_one_epoch.std(dim=0),
            'rej': rej_one_epoch.nanmean(dim=0),
            'rej_err': rej_one_epoch.std(dim=0),
        }
        return metrics
        #return last_loss, acc_one_epoch.nanmean(dim=0), eff_one_epoch.nanmean(dim=0), rej_one_epoch.nanmean(dim=0)

    def train(self, epochs=10, starting_epoch=0, learning_rate=0.001, save_checkpoint=False, checkpoint_path=None,checkpoint_freq=0.3):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(starting_epoch, epochs):
            print(f"At epoch {epoch}")
            self.epochs.append(epoch)
            #train_loss, train_acc, train_eff, train_rej = self.eval_one_epoch()
            train_metrics = self.eval_one_epoch()
            self.model.train(False)
            val_metrics = self.eval_one_epoch(train=False)
            self.train_loss.append(train_metrics['loss'])
            self.train_acc.append(train_metrics['acc'])
            self.train_eff.append(train_metrics['eff'])
            self.train_rej.append(train_metrics['rej'])
            self.train_acc_err.append(train_metrics['acc_err'])
            self.train_eff_err.append(train_metrics['eff_err'])
            self.train_rej_err.append(train_metrics['rej_err'])
            self.val_loss.append(val_metrics['loss'])
            self.val_acc.append(val_metrics['acc'])
            self.val_eff.append(val_metrics['eff'])
            self.val_rej.append(val_metrics['rej'])
            self.val_acc_err.append(val_metrics['acc_err'])
            self.val_eff_err.append(val_metrics['eff_err'])
            self.val_rej_err.append(val_metrics['rej_err'])
            print(f"Train Loss: {train_metrics['loss']:03f}")
            print(f"Val Loss: {val_metrics['loss']:03f}")
            print(f"Train Acc: {train_metrics['acc']} +/- {train_metrics['acc_err']}")
            print(f"Val Acc: {val_metrics['acc']} +/- {val_metrics['acc_err']}")
            print(f"Train Eff: {train_metrics['eff']} +/- {train_metrics['eff_err']}")
            print(f"Val Eff: {val_metrics['eff']} +/- {val_metrics['eff_err']}")
            print(f"Train Rej: {train_metrics['rej']} +/- {train_metrics['rej_err']}")
            print(f"Val Rej: {val_metrics['rej']} +/- {val_metrics['rej_err']}")
            # checkpoint
            if save_checkpoint:
                safe_epoch_frac = int(checkpoint_freq*epochs)
                if safe_epoch_frac == 0:
                    safe_epoch_frac = epochs+1
                if epoch % safe_epoch_frac == 0 and epoch != 0:
                    print(f"Saving checkpoint at epoch {epoch}")
                    # Save the model and other properties
                    file_path=f'{checkpoint_path}checkpoint_{epoch}.pt'
                    self.epoch_warmstart = epoch
                    self.save_checkpoint(file_path=f'{checkpoint_path}checkpoint_{epoch}.pt')



    def save_dataframe(self, file_name):
        data =  {
            "train_loss":self.train_loss,
            "val_loss":self.val_loss,
        }
        for nLCA in range(self.LCA_classes):
            data[f"train_acc_LCA{nLCA}"] = list(np.array(self.train_acc)[:, nLCA])
            data[f"val_acc_LCA{nLCA}"] = list(np.array(self.val_acc)[:, nLCA])
            data[f"train_eff_LCA{nLCA}"] = list(np.array(self.train_eff)[:, nLCA])
            data[f"val_eff_LCA{nLCA}"] = list(np.array(self.val_eff)[:, nLCA])
            data[f"train_rej_LCA{nLCA}"] = list(np.array(self.train_rej)[:, nLCA])
            data[f"val_rej_LCA{nLCA}"] = list(np.array(self.val_rej)[:, nLCA])
            
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
