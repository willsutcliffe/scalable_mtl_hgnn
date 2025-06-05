from wmpgnn.trainers.neutrals_trainer import NeutralsTrainer
from wmpgnn.util.functions import msg, neutrals_hetero_positive_edge_weight, neutrals_hetero_positive_node_weight, weight_binary_class, acc_binary, eff_binary, rej_binary
import torch
from torch import nn
from torch_scatter import scatter_add
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix


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

        self.ce_train_loss = []
        self.ce_val_loss = []
        self.bce_edges_train_loss = []
        self.bce_edges_val_loss = []

        # Metrics tracking

    
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

        if train:
            data_loader = self.train_loader
            # self.model.train()
            # torch.set_grad_enabled(True)
        else:
            data_loader = self.val_loader
            # self.model.eval()
            # self.model.train()
            # torch.set_grad_enabled(False)

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

            ### [DEBUG]
            edge_probs = edge_probs.detach().cpu()

            # Boolean mask of predicted positive edges
            pred_positive = edge_probs > self.threshold
            # label_edges = label_edges.squeeze() # transforms from [N,1] to [N]


            ### [DEBUG]
            label_edges = label_edges.squeeze().detach().cpu() # transforms from [N,1] to [N]

            edge_index = data[('chargedtree', 'to', 'neutrals')].edge_index

            # # --- Aggregate per neutral node: mark neutral as signal if any connecting edge is positive ---
            # num_neutrals = data['neutrals'].num_nodes
            # # For each edge, map to target neutral index
            # neutral_targets = edge_index[1].detach().cpu()
            # # Create tensor of zeros for accumulative signal counts
            # sig_count = pred_positive.new_zeros(num_neutrals, dtype=torch.long)
            # # Scatter add boolean mask (converted to long) to count positives per neutral
            # sig_count = scatter_add(
            #     pred_positive.long(),  # 1 for positive, 0 otherwise
            #     neutral_targets,       # index per edge
            #     dim=0,
            #     out=sig_count
            # )
            # # Build node-level labels: signal if count > 0, else background
            # label_neutrals = (sig_count > 0).long()

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
            # acc_one_batch = acc_binary(pred_positive, label_edges)
            # eff_one_batch = eff_binary(pred_positive, label_edges)
            # rej_one_batch = rej_binary(pred_positive, label_edges)
            # acc_one_epoch.append(acc_one_batch)
            # eff_one_epoch.append(eff_one_batch)
            # rej_one_epoch.append(rej_one_batch)
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

        # Aggregate epoch-wise tensors/lists
        # acc_one_epoch = torch.stack(acc_one_epoch) if acc_one_epoch else torch.tensor([])
        # eff_one_epoch = torch.stack(eff_one_epoch) if eff_one_epoch else torch.tensor([])
        # rej_one_epoch = torch.stack(rej_one_epoch) if rej_one_epoch else torch.tensor([])

        if len(preds_one_epoch) > 0:
            epoch_preds  = torch.cat(preds_one_epoch, dim=0)  # shape [total_edges_in_epoch]
            epoch_labels = torch.cat(labels_one_epoch, dim=0) # même shape
        else:
            # epoch_preds  = torch.tensor([], device='cuda')
            # epoch_labels = torch.tensor([], device='cuda')

            ## [DEBUG]
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
            # 'acc': acc_one_epoch.nanmean(dim=0),
            # 'acc_err': acc_one_epoch.std(dim=0),
            # 'eff': eff_one_epoch.nanmean(dim=0),
            # 'eff_err': eff_one_epoch.std(dim=0),
            # 'rej': rej_one_epoch.nanmean(dim=0),
            # 'rej_err': rej_one_epoch.std(dim=0),
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
            self.val_predictions.append(val_metrics['preds'])
            self.val_labels.append(val_metrics['labels'])
            self.val_loss.append(val_metrics['loss'])

            # --- Compute thresholds and per-threshold metrics for TRAIN and VAL ---
            train_preds_np = train_metrics['preds'].numpy()
            train_labels_np = train_metrics['labels'].numpy()
            val_preds_np   = val_metrics['preds'].numpy()
            val_labels_np  = val_metrics['labels'].numpy()

            train_dict = self.compute_thresholds_and_metrics(
                train_labels_np, train_preds_np, key_prefix='train', epoch=epoch
            )
            val_dict   = self.compute_thresholds_and_metrics(
                val_labels_np,   val_preds_np,   key_prefix='val', epoch=epoch
            )

            # Merge the two dicts to form this epoch's row
            epoch_metric_dict = {**train_dict, **val_dict}
            epoch_series = pd.Series(epoch_metric_dict, name=epoch)
            self.epoch_metrics_df = pd.concat(
                [self.epoch_metrics_df, epoch_series.to_frame().T],
                axis=0
            )

            # --- Print metrics (manual threshold) via get_epoch_metric ---
            tm_acc  = self.get_epoch_metric('train_manual_accuracy', epoch=epoch)
            vm_acc  = self.get_epoch_metric('val_manual_accuracy', epoch=epoch)
            tm_tpr  = self.get_epoch_metric('train_manual_TPR', epoch=epoch)
            vm_tpr  = self.get_epoch_metric('val_manual_TPR', epoch=epoch)
            tm_rej  = self.get_epoch_metric('train_manual_rej', epoch=epoch)
            vm_rej  = self.get_epoch_metric('val_manual_rej', epoch=epoch)

            print(f"Epoch {epoch} | Manual threshold:")
            print(f"  Train - Acc: {tm_acc:.4f}, Eff: {tm_tpr:.4f}, Rej: {tm_rej:.4f}")
            print(f"  Val   - Acc: {vm_acc:.4f}, Eff: {vm_tpr:.4f}, Rej: {vm_rej:.4f}")

           
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


    def compute_thresholds_and_metrics(self, y_true: np.ndarray, y_score: np.ndarray, key_prefix: str, epoch=-1):
        """
        Given true labels (0/1) and scores (float) for the entire epoch,
        compute for 4 thresholds (manuel, opt, TPR=0.9, TPR=0.99) :
          - confusion matrix (TP, FP, TN, FN)
          - TPR (efficiency), Rejection (1 - FPR), Precision, Accuracy, Balanced accuracy
        key_prefix = 'train' or 'val' to differentiate keys in the final DataFrame.
        Returns a dict whose keys are, for example:
          "train_manual_TP", "train_manual_FP", "train_manual_TN", "train_manual_FN",
          "train_manual_TPR", "train_manual_rej", "train_manual_precision",
          "train_manual_accuracy", "train_manual_balanced_accuracy",
          "train_manual_threshold_value", and similarly for 'opt', 'tpr0.9', 'tpr0.99'.
        """
        metrics_dict = {}

        # Ensure labels are int
        y_true = y_true.astype(int)

        # Compute ROC curve (fpr, tpr, thresholds)
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        thresholds = thresholds[1:]
        fpr = fpr[1:]
        tpr = tpr[1:]

        # Number of signal/background
        N_signal = int((y_true == 1).sum())
        N_background = int((y_true == 0).sum())

        # Compute S and B for each threshold
        S_arr = tpr * N_signal
        B_arr = fpr * N_background
        fom = np.divide(S_arr, np.sqrt(S_arr + B_arr), out=np.zeros_like(S_arr), where=(S_arr + B_arr) > 0)
        opt_idx = np.nanargmax(fom)
        opt_threshold = thresholds[opt_idx]
        tpr_at_opt = tpr[opt_idx]


        # Helper to find the largest threshold that yields tpr >= target
        def find_threshold_for_tpr(target_tpr):
            idxs = np.where(tpr >= target_tpr)[0]
            if idxs.size == 0:
                # If no threshold reaches that TPR, pick the smallest threshold
                return thresholds[-1]
            else:
                # Return the last threshold in thresholds where tpr >= target_tpr
                return thresholds[idxs[0]]

        tpr09_threshold = find_threshold_for_tpr(0.9)
        tpr099_threshold = find_threshold_for_tpr(0.99)

        threshold_info = [
            ('manual', self.threshold),
            ('opt',    opt_threshold),
            ('tpr0.9', tpr09_threshold),
            ('tpr0.99',tpr099_threshold),
        ]

        total_samples = y_true.shape[0]

        for name, thr in threshold_info:
            # Binarize predictions at this threshold
            y_pred_bin = (y_score > thr).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin, labels=[0, 1]).ravel()

            # TPR = TP / (TP + FN)
            tpr_val  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            # Rejection = TN / (TN + FP)
            rej_val  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            # Precision = TP / (TP + FP)
            prec_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            # Accuracy = (TP + TN) / total_samples
            acc_val  = (tp + tn) / total_samples if total_samples > 0 else 0.0
            # Balanced accuracy = 0.5 * (TPR + TN/(TN + FP))
            bal_acc_val = 0.5 * (tpr_val + rej_val)

            prefix = f"{key_prefix}_{name}"
            metrics_dict[f"{prefix}_TP"]                = int(tp)
            metrics_dict[f"{prefix}_FP"]                = int(fp)
            metrics_dict[f"{prefix}_TN"]                = int(tn)
            metrics_dict[f"{prefix}_FN"]                = int(fn)
            metrics_dict[f"{prefix}_TPR"]               = float(tpr_val)
            metrics_dict[f"{prefix}_rej"]               = float(rej_val)
            metrics_dict[f"{prefix}_precision"]         = float(prec_val)
            metrics_dict[f"{prefix}_accuracy"]          = float(acc_val)
            metrics_dict[f"{prefix}_balanced_accuracy"] = float(bal_acc_val)

        # Also store the numeric threshold values themselves
        metrics_dict[f"{key_prefix}_manual_threshold_value"]   = float(self.threshold)
        metrics_dict[f"{key_prefix}_opt_threshold_value"]      = float(opt_threshold)
        metrics_dict[f"{key_prefix}_tpr0.9_threshold_value"]  = float(tpr09_threshold)
        metrics_dict[f"{key_prefix}_tpr0.99_threshold_value"] = float(tpr099_threshold)
        
        self.tpr_and_threshold[key_prefix][epoch] = {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'threshold_manual': self.threshold,
            'threshold_opt': opt_threshold,
            'tpr_at_opt': tpr_at_opt,
            'threshold_tpr_90': tpr09_threshold,
            'threshold_tpr_99': tpr099_threshold,
            'fom': fom
        }
        return metrics_dict

    def save_metrics(self, file_name: str):
        """
        Save the per-epoch metrics DataFrame to CSV and return le DataFrame.
        """
        self.epoch_metrics_df.to_csv(file_name, index_label='epoch')
        return self.epoch_metrics_df

    def get_epoch_metric(self, column_name: str, epoch=None):
        """
        If epoch is None, returns the entire column as a NumPy array.
        Otherwise, returns the single value at (epoch, column_name).
        """
        if column_name not in self.epoch_metrics_df.columns:
            raise KeyError(f"Column '{column_name}' not found in epoch_metrics_df.")

        if epoch is None:
            # Return the whole column
            return self.epoch_metrics_df[column_name].values
        else:
            if epoch in self.epoch_metrics_df.index:
                return self.epoch_metrics_df.loc[epoch, column_name]
            else:
                raise KeyError(f"Epoch {epoch} not found in epoch_metrics_df.")