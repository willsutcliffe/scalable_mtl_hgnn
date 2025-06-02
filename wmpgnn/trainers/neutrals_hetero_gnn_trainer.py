from wmpgnn.trainers.neutrals_trainer import NeutralsTrainer
from wmpgnn.util.functions import msg, neutrals_hetero_positive_edge_weight, neutrals_hetero_positive_node_weight, weight_binary_class, acc_binary, eff_binary, rej_binary
import torch
from torch import nn
from torch_scatter import scatter_add
import numpy as np
import pandas as pd

class NeutralsHeteroGNNTrainer(NeutralsTrainer):
    """ Class for training """

    def __init__(self, config, model, train_loader, val_loader, add_bce=True,
                use_bce_pos_weight=False, threshold=0.5):
        super().__init__(config, model, train_loader, val_loader)
        self.threshold=threshold
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        # Class weighting for the classification task
        weights = weight_binary_class(self.train_loader, hetero=True)
        pos_weight = (weights[1] / weights[0]).clone().detach().cuda()

        # Initialize the criterion with pos_weight as computed
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # BCE loss setup for edge prediction
        if use_bce_pos_weight:
            pos_weight = neutrals_hetero_positive_edge_weight(train_loader)
            pos_weight = torch.tensor([pos_weight])
            self.criterion_bce_edges = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            self.use_logits = True
        else:
            self.criterion_bce_edges = nn.BCELoss() ### use this one
            self.use_logits = False

        print("Use logits", self.use_logits)

        self.criterion.to('cuda')
        self.criterion_bce_edges.cuda()
        self.model.cuda()

        self.add_bce = add_bce
        self.beta_bce_edges = 33.2256  # You may want to recompute for new setup

        # Metrics tracking
        self.ce_train_loss = []
        self.ce_val_loss = []
        self.bce_edges_train_loss = []
        self.bce_edges_val_loss = []

    
    def save_checkpoint(self,file_path:str):
        """Saves the model and optimizer state to a checkpoint file."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'criterion_state_dict': self.criterion.state_dict(),
            # 'criterion_bce_nodes_state_dict': self.criterion_bce_nodes.state_dict(),
            'criterion_bce_edges_state_dict': self.criterion_bce_edges.state_dict(),
            # 'criterion_bce_pvs_state_dict': self.criterion_bce_pvs.state_dict(),
            'epoch_warmstart': self.epoch_warmstart,
            'history': self.get_history(),
        }
        torch.save(checkpoint, file_path)
        print(f"Checkpoint saved to {file_path}")
    
    def load_checkpoint(self, file_path=None):
        """Loads the model and optimizer state from a checkpoint file."""
        checkpoint = torch.load(file_path, weights_only=True)

        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'criterion_state_dict' in checkpoint:
            self.criterion.load_state_dict(checkpoint['criterion_state_dict'])
        if 'criterion_bce_edges_state_dict' in checkpoint:
            self.criterion_bce_edges.load_state_dict(checkpoint['criterion_bce_edges_state_dict'])
        if 'history' in checkpoint:
            self.set_history(checkpoint['history'])
        if 'epoch_warmstart' in checkpoint:
            self.epoch_warmstart = checkpoint['epoch_warmstart'] + 1


    def get_history(self):
        """Returns the training and validation history of the heterogeneous model's metrics"""
        history = super().get_history()
        # Retain only edge-level losses and metrics for chargedtree -> neutrals
        if self.add_bce:
            history['bce_edges_train_loss'] = self.bce_edges_train_loss
            history['bce_edges_val_loss']   = self.bce_edges_val_loss
        return history

    def set_history(self, history):
        """Set the training and validation history of the heterogeneous model's metrics"""
        super().set_history(history)
        # Restore only edge-level losses
        if self.add_bce:
            self.bce_edges_train_loss = history.get('bce_edges_train_loss', [])
            self.bce_edges_val_loss   = history.get('bce_edges_val_loss', [])
        
    def set_beta_BCE_nodes(self, beta):
        self.beta_BCE_nodes = beta

    def set_beta_BCE_edges(self, beta):
        self.beta_BCE_edges = beta

    def set_beta_BCE_pvs(self, beta):
        self.beta_BCE_pvs= beta

    def eval_one_epoch(self, train=True):
        running_loss = 0.
        last_loss = 0.
        running_ce_loss = 0.
        running_bce_edge_loss = 0.
        acc_one_epoch = []
        eff_one_epoch = []
        rej_one_epoch = []
        preds_one_epoch =[]
        labels_one_epoch = []

        if train == True:
            data_loader = self.train_loader
            self.model.train()
            torch.set_grad_enabled(True)
        else:
            data_loader = self.val_loader
            self.model.eval()
            torch.set_grad_enabled(False)

        last_batch = len(data_loader)
        # print(last_batch)

        # with torch.no_grad():
        for i, data in enumerate(data_loader):
            # Zero gradients if training
            if train:
                self.optimizer.zero_grad()
            data.to('cuda')

            # Forward pass through the model
            outputs = self.model(data)
            data = outputs

            # --- Edge classification loss for chargedtree -> neutrals ---
            # True labels: 0 = background, 1 = signal
            # y is one-hot or logits over two classes
            label_edges = data[('chargedtree', 'to', 'neutrals')].y
            # Compute cross-entropy loss on edges

            loss = self.criterion(
                outputs[('chargedtree', 'to', 'neutrals')].edges,
                label_edges
            )
            running_ce_loss += loss.item()

            # --- Determine which edges are predicted positive (signal) ---
            edge_probs = torch.sigmoid(
                outputs[('chargedtree', 'to', 'neutrals')].edges
            )[:,0]
            edge_probs = edge_probs.detach().cpu()
            # Boolean mask of predicted positive edges
            pred_positive = edge_probs > self.threshold
            label_edges = label_edges.squeeze().detach().cpu() # transforms from [N,1] to [N]

            edge_index = data[('chargedtree', 'to', 'neutrals')].edge_index

            # --- Aggregate per neutral node: mark neutral as signal if any connecting edge is positive ---
            num_neutrals = data['neutrals'].num_nodes
            # For each edge, map to target neutral index
            neutral_targets = edge_index[1]
            # Create tensor of zeros for accumulative signal counts
            sig_count = pred_positive.new_zeros(num_neutrals, dtype=torch.long)
            # Scatter add boolean mask (converted to long) to count positives per neutral
            sig_count = scatter_add(
                pred_positive.long(),  # 1 for positive, 0 otherwise
                neutral_targets,       # index per edge
                dim=0,
                out=sig_count
            )
            # Build node-level labels: signal if count > 0, else background
            label_neutrals = (sig_count > 0).long()

            ## Compute additional losses and metrics for the new edge type
            for block in self.model._blocks:
                if self.use_logits:
                    if self.add_bce:
                        # BCE loss on ('chargedtree', 'to', 'neutrals') edge logits
                        bce_edges_loss = self.beta_bce_edges * self.criterion_bce_edges(
                            block.edge_logits[('chargedtree', 'to', 'neutrals')],
                            data[('chargedtree', 'to', 'neutrals')].y.float()  # Assuming binary class (0/1)
                        )
                        running_bce_edge_loss += bce_edges_loss.item()
                        loss += bce_edges_loss
                else:
                    if self.add_bce:
                        # BCE loss on weights instead of logits
                        bce_edges_loss = self.beta_bce_edges * self.criterion_bce_edges(
                            block.edge_weights[('chargedtree', 'to', 'neutrals')],
                            data[('chargedtree', 'to', 'neutrals')].y.float()
                        )
                        running_bce_edge_loss += bce_edges_loss.item()
                        loss += bce_edges_loss

            # Accuracy/effectiveness/rejection metrics for new edge type
            acc_one_batch = acc_binary(pred_positive, label_edges)
            eff_one_batch = eff_binary(pred_positive, label_edges)
            rej_one_batch = rej_binary(pred_positive, label_edges)
            acc_one_epoch.append(acc_one_batch)
            eff_one_epoch.append(eff_one_batch)
            rej_one_epoch.append(rej_one_batch)
            preds_one_epoch.append(edge_probs)
            labels_one_epoch.append(label_edges)

            # For training only
            if train :
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()
            if (i + 1) == last_batch:
                last_loss = running_loss / last_batch  # loss per batch
                info_msg = '  batch {} last_batch {} loss: {}'.format(i + 1, last_batch, last_loss)
                print(info_msg)
                running_loss = 0.

        acc_one_epoch = torch.stack(acc_one_epoch)
        eff_one_epoch = torch.stack(eff_one_epoch)
        rej_one_epoch = torch.stack(rej_one_epoch)

        if len(preds_one_epoch) > 0:
            epoch_preds  = torch.cat(preds_one_epoch, dim=0)  # shape [total_edges_in_epoch]
            epoch_labels = torch.cat(labels_one_epoch, dim=0) # même shape
        else:
            # epoch_preds  = torch.tensor([], device='cuda')
            # epoch_labels = torch.tensor([], device='cuda')
            epoch_preds  = torch.tensor([], dtype=torch.float32)
            epoch_labels = torch.tensor([], dtype=torch.long)

        preds_one_epoch.clear()
        labels_one_epoch.clear()
        del data
        del outputs
        torch.cuda.empty_cache()

        if train:
            self.ce_train_loss.append(running_ce_loss / last_batch)
            self.bce_edges_train_loss.append(running_bce_edge_loss / last_batch)
            # self.bce_nodes_train_loss.append(running_bce_node_loss / last_batch)
            # self.bce_pvs_train_loss.append(running_bce_pv_loss / last_batch)
        else:
            self.ce_val_loss.append(running_ce_loss / last_batch)
            self.bce_edges_val_loss.append(running_bce_edge_loss / last_batch)
            # self.bce_nodes_val_loss.append(running_bce_node_loss / last_batch)
            # self.bce_pvs_val_loss.append(running_bce_pv_loss / last_batch)

        metrics = {
            'preds' : epoch_preds,
            'labels': epoch_labels,
            'loss': last_loss,
            'acc': acc_one_epoch.nanmean(dim=0),
            'acc_err': acc_one_epoch.std(dim=0),
            'eff': eff_one_epoch.nanmean(dim=0),
            'eff_err': eff_one_epoch.std(dim=0),
            'rej': rej_one_epoch.nanmean(dim=0),
            'rej_err': rej_one_epoch.std(dim=0),
        }

        return metrics

    def train(self, epochs=10, starting_epoch=0, learning_rate=0.001,
              save_checkpoint=False, checkpoint_path=None, checkpoint_freq=0.3):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(starting_epoch, epochs):
            msg(f"At epoch {epoch}")
            self.epochs.append(epoch)

            # Training epoch
            train_metrics = self.eval_one_epoch(train=True)
            self.model.train(False)
            # Validation epoch
            val_metrics = self.eval_one_epoch(train=False)

            # Append metrics
            self.train_predictions.append(train_metrics['preds'])
            self.train_labels.append(train_metrics['labels'])

            self.train_loss.append(train_metrics['loss'])
            self.train_acc.append(train_metrics['acc'])
            self.train_eff.append(train_metrics['eff'])
            self.train_rej.append(train_metrics['rej'])
            self.train_acc_err.append(train_metrics['acc_err'])
            self.train_eff_err.append(train_metrics['eff_err'])
            self.train_rej_err.append(train_metrics['rej_err'])

            self.val_predictions.append(val_metrics['preds'])
            self.val_labels.append(val_metrics['labels'])

            self.val_loss.append(val_metrics['loss'])
            self.val_acc.append(val_metrics['acc'])
            self.val_eff.append(val_metrics['eff'])
            self.val_rej.append(val_metrics['rej'])
            self.val_acc_err.append(val_metrics['acc_err'])
            self.val_eff_err.append(val_metrics['eff_err'])
            self.val_rej_err.append(val_metrics['rej_err'])

            # Logging
            print(f"Train Loss: {train_metrics['loss']:.6f}")
            print(f"Val Loss: {val_metrics['loss']:.6f}")
            print(f"Train Acc: {train_metrics['acc']} +/- {train_metrics['acc_err']}")
            print(f"Val Acc: {val_metrics['acc']} +/- {val_metrics['acc_err']}")
            print(f"Train Eff: {train_metrics['eff']} +/- {train_metrics['eff_err']}")
            print(f"Val Eff: {val_metrics['eff']} +/- {val_metrics['eff_err']}")
            print(f"Train Rej: {train_metrics['rej']} +/- {train_metrics['rej_err']}")
            print(f"Val Rej: {val_metrics['rej']} +/- {val_metrics['rej_err']}")

            # Checkpoint saving
            if save_checkpoint:
                safe_epoch_frac = int(checkpoint_freq * epochs)
                if safe_epoch_frac == 0:
                    safe_epoch_frac = epochs + 1
                if epoch % safe_epoch_frac == 0 and epoch != 0:
                    print(f"Saving checkpoint at epoch {epoch}")
                    self.epoch_warmstart = epoch
                    file_path = f'{checkpoint_path}checkpoint_{epoch}.pt'
                    self.save_checkpoint(file_path)

        # self.train_predicitions = torch.cat(self.train_predictions)
        # self.val_predicitions = torch.cat(self.val_predictions)
        # self.train_labels = torch.cat(self.train_labels)
        # self.val_labels = torch.cat(self.val_labels)


    def save_dataframe(self, file_name):
        data = {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
        }
        # Metrics per class for edge classification
        data["train_acc"] = [x.cpu().item() if torch.is_tensor(x) else float(x) for x in self.train_acc]
        data["val_acc"] = [x.cpu().item() if torch.is_tensor(x) else float(x) for x in self.val_acc]
        data["train_eff"] = [x.cpu().item() if torch.is_tensor(x) else float(x) for x in self.train_eff]
        data["val_eff"] = [x.cpu().item() if torch.is_tensor(x) else float(x) for x in self.val_eff]
        data["train_rej"] = [x.cpu().item() if torch.is_tensor(x) else float(x) for x in self.train_rej]
        data["val_rej"] = [x.cpu().item() if torch.is_tensor(x) else float(x) for x in self.val_rej]

        
        if self.add_bce:
            data["ce_train_loss"] = self.ce_train_loss
            data["ce_val_loss"]  = self.ce_val_loss
            # data["bce_nodes_train_loss"] = self.bce_nodes_train_loss
            # data["bce_nodes_val_loss"] = self.bce_nodes_val_loss
            data["bce_edges_train_loss"]  = self.bce_edges_train_loss
            data["bce_edges_val_loss"] = self.bce_edges_val_loss

        df = pd.DataFrame(data)
        df.to_csv(file_name)
        return df


    # def save_dataframe(self, file_name):
    #     data = {
    #         "train_loss": self.train_loss,
    #         "val_loss": self.val_loss,
    #     }

    #     def safe_extract(metric_list, index):
    #         values = []
    #         for x in metric_list:
    #             if torch.is_tensor(x):
    #                 x = x.cpu()
    #                 if x.ndim == 0:
    #                     # Scalar tensor
    #                     values.append(x.item())
    #                 else:
    #                     # Vector tensor
    #                     values.append(x[index].item())
    #             else:
    #                 # Not a tensor (e.g., float or list)
    #                 if isinstance(x, (list, tuple, np.ndarray)):
    #                     values.append(x[index])
    #                 else:
    #                     values.append(x)
    #         return values

    #     for nClasse in range(self.neutrals_classes):
    #         data[f"train_acc_Class{nClasse}"] = safe_extract(self.train_acc, nClasse)
    #         data[f"val_acc_Class{nClasse}"] = safe_extract(self.val_acc, nClasse)
    #         data[f"train_eff_Class{nClasse}"] = safe_extract(self.train_eff, nClasse)
    #         data[f"val_eff_Class{nClasse}"] = safe_extract(self.val_eff, nClasse)
    #         data[f"train_rej_Class{nClasse}"] = safe_extract(self.train_rej, nClasse)
    #         data[f"val_rej_Class{nClasse}"] = safe_extract(self.val_rej, nClasse)

    #     if self.add_bce:
    #         data["ce_train_loss"] = [x.cpu().item() if torch.is_tensor(x) else x for x in self.ce_train_loss]
    #         data["ce_val_loss"]  = [x.cpu().item() if torch.is_tensor(x) else x for x in self.ce_val_loss]
    #         data["bce_edges_train_loss"] = [x.cpu().item() if torch.is_tensor(x) else x for x in self.bce_edges_train_loss]
    #         data["bce_edges_val_loss"]   = [x.cpu().item() if torch.is_tensor(x) else x for x in self.bce_edges_val_loss]

    #     df = pd.DataFrame(data)
    #     df.to_csv(file_name)
    #     return df

