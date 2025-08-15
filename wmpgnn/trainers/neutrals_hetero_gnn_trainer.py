from wmpgnn.trainers.neutrals_trainer import NeutralsTrainer
from wmpgnn.util.functions import msg, neutrals_hetero_positive_edge_weight, neutrals_hetero_positive_node_weight, weight_binary_class, acc_binary, eff_binary, rej_binary
import torch
from torch import nn
from torch_scatter import scatter_add
from torch.utils.data import SubsetRandomSampler, DataLoader
import numpy as np
import pandas as pd
import copy
from sklearn.metrics import roc_curve, auc, confusion_matrix


class NeutralsHeteroGNNTrainer(NeutralsTrainer):
    """
    Trainer for heterogeneous GNNs for neutral inclusion:
      - Binary edge classification (neutral inclusion task)

    Inherits from:
        Trainer: abstract base class for training loops.
    """
    def __init__(self, config, model, train_loader, val_loader, add_bce=True,
                use_bce_pos_weight=False, threshold=0.5):
        """
        Initialize the NuetralsHeteroGNNTrainer.

        Args:
            config (dict): Configuration dict (must include 'device').
            model (nn.Module): Neutrals Heterogeneous GNN model.
            train_loader: DataLoader for training graphs.
            val_loader: DataLoader for validation graphs.
            add_bce (bool): Include BCE losses for edges/nodes.
            use_bce_pos_weight (bool): Use positive-class weighting in BCEWithLogitsLoss.
            threshold: Set the value to separate background and signal
        """
        super().__init__(config, model, train_loader, val_loader)
        self.threshold = threshold
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4)
        self.k_subsetRandomSampler = config.get("training.k_subsetRandomSampler")
        self._base_train_loader = self.train_loader

        # Compute class weights for binary classification on edges
        weights = weight_binary_class(self.train_loader, hetero=True)
        pos_weight = (weights[1] / weights[0]).clone().detach().cuda()

        # Initialize the main criterion for BCE loss with logits and class weighting
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Setup BCE loss criterion for edge predictions
        if use_bce_pos_weight:
            pos_weight = neutrals_hetero_positive_edge_weight(train_loader)
            pos_weight = torch.tensor([pos_weight])
            self.criterion_bce_edges = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            self.use_logits = True
        else:
            self.criterion_bce_edges = nn.BCELoss()  # Use standard BCE loss without logits
            self.use_logits = False

        print("Use logits", self.use_logits)

        # Move loss functions and model to GPU
        self.criterion.to('cuda')
        self.criterion_bce_edges.cuda()
        self.model.cuda()

        self.add_bce = add_bce
        self.beta_bce_edges = 33.2256  # Scaling factor for BCE edge loss; adjust if needed

        # Lists to store loss history for training and validation
        self.ce_train_loss = []
        self.ce_val_loss = []
        self.bce_edges_train_loss = []
        self.bce_edges_val_loss = []

        # Track last epoch number for early stopping or resuming
        self.last_epoch = 0

    def save_checkpoint(self, file_path: str):
        """Save model, optimizer, and loss state to a checkpoint file."""
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
        """Load model, optimizer, and loss state from a checkpoint file."""
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
        """Retrieve training and validation loss history including edge BCE losses."""
        history = super().get_history()
        if self.add_bce:
            history['bce_edges_train_loss'] = self.bce_edges_train_loss
            history['bce_edges_val_loss'] = self.bce_edges_val_loss
        return history

    def set_history(self, history):
        """Set training and validation loss history, restoring edge BCE losses if present."""
        super().set_history(history)
        if self.add_bce:
            self.bce_edges_train_loss = history.get('bce_edges_train_loss', [])
            self.bce_edges_val_loss = history.get('bce_edges_val_loss', [])
        
    def set_beta_BCE_nodes(self, beta):
        """Set scaling factor for node-level BCE loss."""
        self.beta_BCE_nodes = beta

    def set_beta_BCE_edges(self, beta):
        """Set scaling factor for edge-level BCE loss."""
        self.beta_BCE_edges = beta

    def set_beta_BCE_pvs(self, beta):
        """Set scaling factor for PV-association BCE loss."""
        self.beta_BCE_pvs = beta

    def eval_one_epoch(self, train=True):
        """
        Evaluate one epoch of training or validation.
        
        Args:
            train: If True, run training step (with backprop). If False, run validation.
        
        Returns:
            metrics: Dictionary with predictions, labels, and loss for the epoch.
        """
        running_loss = 0.
        last_loss = 0.
        running_ce_loss = 0.
        running_bce_edge_loss = 0.
        acc_one_epoch = []
        eff_one_epoch = []
        rej_one_epoch = []
        preds_one_epoch = []
        labels_one_epoch = []
        neutrals_id_one_epoch = []

        data_loader = self.train_loader if train else self.val_loader

        last_batch = len(data_loader)

        # Loop over batches
        for i, data in enumerate(data_loader):
            if train:
                self.optimizer.zero_grad()
            data.to('cuda')

            # Forward pass through the model
            outputs = self.model(data)
            data = outputs

            # Binary classification loss on edges from chargedtree to neutrals
            label_edges = data[('chargedtree', 'to', 'neutrals')].y

            ### DEBUG TODO !!!!
            neutral_id_edges = data[('chargedtree', 'to', 'neutrals')].neutrals_id
            # neutral_id_edges = data[('chargedtree', 'to', 'neutrals')].y

            loss = self.criterion(
                outputs[('chargedtree', 'to', 'neutrals')].edges,
                label_edges
            )
            running_ce_loss += loss.item()

            # Compute predicted edge probabilities by applying sigmoid
            edge_probs = torch.sigmoid(
                outputs[('chargedtree', 'to', 'neutrals')].edges
            )[:, 0]

            # Move to CPU for analysis
            edge_probs = edge_probs.detach().cpu()

            # Determine which edges are predicted positive by thresholding
            pred_positive = edge_probs > self.threshold

            # Squeeze labels to 1D tensor and move to CPU
            label_edges = label_edges.squeeze().detach().cpu()
            neutral_id_edges = neutral_id_edges.squeeze().detach().cpu()
            edge_index = data[('chargedtree', 'to', 'neutrals')].edge_index

            # Compute additional BCE loss on edge logits or weights if configured
            for block in self.model._blocks:
                if self.use_logits:
                    if self.add_bce:
                        bce_edges_loss = self.beta_bce_edges * self.criterion_bce_edges(
                            block.edge_logits[('chargedtree', 'to', 'neutrals')],
                            data[('chargedtree', 'to', 'neutrals')].y.float()
                        )
                        running_bce_edge_loss += bce_edges_loss.item()
                        loss += bce_edges_loss
                else:
                    if self.add_bce:
                        bce_edges_loss = self.beta_bce_edges * self.criterion_bce_edges(
                            block.edge_weights[('chargedtree', 'to', 'neutrals')],
                            data[('chargedtree', 'to', 'neutrals')].y.float()
                        )
                        running_bce_edge_loss += bce_edges_loss.item()
                        loss += bce_edges_loss

            # Store predictions and labels for the epoch
            preds_one_epoch.append(edge_probs)
            labels_one_epoch.append(label_edges)
            neutrals_id_one_epoch.append(neutral_id_edges)

            if train:
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()

            # At last batch, compute average loss and print info
            if (i + 1) == last_batch:
                last_loss = running_loss / last_batch
                info_msg = f'  batch {i + 1} last_batch {last_batch} loss: {last_loss}'
                print(info_msg)
                running_loss = 0.

        # Concatenate predictions and labels from all batches
        if len(preds_one_epoch) > 0:
            epoch_preds = torch.cat(preds_one_epoch, dim=0)
            epoch_labels = torch.cat(labels_one_epoch, dim=0)
            epoch_neutrals_id = torch.cat(neutrals_id_one_epoch, dim=0)
        else:
            epoch_preds = torch.tensor([], dtype=torch.float32)
            epoch_labels = torch.tensor([], dtype=torch.long)
            epoch_neutrals_id = torch.tensor([], dtype=torch.float32)


        preds_one_epoch.clear()
        labels_one_epoch.clear()
        neutrals_id_one_epoch.clear()
        del data
        del outputs
        torch.cuda.empty_cache()

        # Append losses to history
        if train:
            self.ce_train_loss.append(running_ce_loss / last_batch)
            self.bce_edges_train_loss.append(running_bce_edge_loss / last_batch)
        else:
            self.ce_val_loss.append(running_ce_loss / last_batch)
            self.bce_edges_val_loss.append(running_bce_edge_loss / last_batch)

        metrics = {
            'preds': epoch_preds,
            'labels': epoch_labels,
            'neutrals_id': epoch_neutrals_id,
            'loss': last_loss,
            # Accuracy, efficiency, rejection metrics are commented out
        }

        return metrics

    def train(self, epochs=10, starting_epoch=0, learning_rate=0.001, early_stopping_patience=100, min_delta=0,
            save_checkpoint=False, checkpoint_path=None, checkpoint_freq=0.3):
        """
        Train the model for a given number of epochs with optional early stopping and checkpoint saving.

        Args:
            epochs (int): Total number of epochs to train.
            starting_epoch (int): Epoch number to start training from (useful for warm restarts).
            learning_rate (float): Learning rate for the Adam optimizer.
            early_stopping_patience (int): Number of epochs to wait without improvement before stopping early.
            min_delta (float): Minimum change in validation loss to qualify as an improvement.
            save_checkpoint (bool): Whether to save model checkpoints during training.
            checkpoint_path (str): Directory path to save checkpoints.
            checkpoint_freq (float): Fraction of total epochs after which to save a checkpoint (e.g. 0.3 means every 30% epochs).

        Workflow:
            - Use Adam optimizer.
            - Support k-fold-like subsampling during training if k_subsetRandomSampler > 1.
            - Evaluate on train and validation set every epoch.
            - Track best validation loss for early stopping.
            - Save checkpoints at specified frequency.
            - Restore best model weights at the end.
        """


        # Initialize optimizer with Adam and the given learning rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        full_dataset = self.train_loader.dataset
        batch_size = self.train_loader.batch_size
        collate_fn = getattr(self.train_loader, "collate_fn", None)
        num_workers = getattr(self.train_loader, "num_workers", 0)

        best_val_loss = float('inf')  # Best validation loss observed so far
        best_model_state = None       # Stores model parameters at best validation loss
        best_epoch = -1               # Epoch number where best model was found
        patience_counter = 0          # Counts epochs without improvement for early stopping

        for epoch in range(starting_epoch, epochs):
            msg(f"At epoch {epoch}")
            self.epochs.append(epoch)

            # --- TRAINING PHASE ---
            if self.k_subsetRandomSampler == 1:
                # If no subsampling, use full train_loader for training
                train_metrics = self.eval_one_epoch(train=True)
            else:
                # Subsampling approach for training data

                # 1) Generate a random permutation of all indices in the dataset
                num_samples = len(full_dataset)
                indices = np.arange(num_samples)
                np.random.shuffle(indices)

                # 2) Exclude approximately 1/k_subsetRandomSampler fraction of the data
                exclude_size = num_samples // self.k_subsetRandomSampler
                included_indices = indices[exclude_size:]  # keep this subset for training

                # 3) Create a SubsetRandomSampler with the included indices
                train_sampler = SubsetRandomSampler(included_indices)

                # 4) Create a temporary DataLoader for this subsample
                train_loader_epoch = DataLoader(
                    full_dataset,
                    batch_size=batch_size,
                    sampler=train_sampler,
                    shuffle=False,       # sampler shuffles data already
                    num_workers=num_workers,
                    collate_fn=collate_fn
                )

                # 5) Temporarily replace self.train_loader by the subsampled loader
                original_loader = self.train_loader
                self.train_loader = train_loader_epoch

                # 6) Perform training epoch using the subsampled loader
                train_metrics = self.eval_one_epoch(train=True)

                # 7) Restore original train_loader for next epochs/validation
                self.train_loader = original_loader

            self.model.train(False)  # Switch model to eval mode for validation

            # Validation phase
            val_metrics = self.eval_one_epoch(train=False)

            # Append predictions, labels, and losses for training and validation
            self.train_predictions.append(train_metrics['preds'])
            self.train_labels.append(train_metrics['labels'])
            self.train_loss.append(train_metrics['loss'])
            self.train_neutrals_id.append(train_metrics['neutrals_id'])
            self.val_predictions.append(val_metrics['preds'])
            self.val_labels.append(val_metrics['labels'])
            self.val_loss.append(val_metrics['loss'])
            self.val_neutrals_id.append(val_metrics['neutrals_id'])


            # Convert tensors to numpy arrays for metric computations
            train_preds_np = train_metrics['preds'].numpy()
            train_labels_np = train_metrics['labels'].numpy()
            val_preds_np   = val_metrics['preds'].numpy()
            val_labels_np  = val_metrics['labels'].numpy()
            train_loss = train_metrics['loss']
            val_loss = val_metrics['loss']
            train_neutrals_id_np=train_metrics['neutrals_id'].numpy()
            val_neutrals_id_np=val_metrics['neutrals_id'].numpy()


            # --- EARLY STOPPING LOGIC ---
            if val_loss is None:
                raise ValueError("Validation loss metric not found in val_metrics")

            # Check if validation loss improved sufficiently
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_epoch = epoch
                best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0  # reset patience counter
            else:
                if val_loss < best_val_loss:
                    # Slight improvement but less than min_delta, increment patience partially
                    best_val_loss = val_loss
                    best_epoch = epoch
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    patience_counter += 0.5
                    print(f"Not enough improvement in validation loss (less than {min_delta}). Patience: {patience_counter}/{early_stopping_patience}")
                else:
                    # No improvement, increment patience fully
                    patience_counter += 1
                    print(f"No improvement in validation loss. Patience: {patience_counter}/{early_stopping_patience}")

            # Compute threshold-dependent metrics for train and val sets
            train_dict = self.compute_thresholds_and_metrics(
                train_labels_np, train_preds_np, train_loss, train_neutrals_id_np, key_prefix='train', epoch=epoch
            )
            val_dict = self.compute_thresholds_and_metrics(
                val_labels_np, val_preds_np, val_loss, val_neutrals_id_np,key_prefix='val', epoch=epoch
            )

            # Merge train and validation metrics for this epoch
            epoch_metric_dict = {**train_dict, **val_dict}
            epoch_series = pd.Series(epoch_metric_dict, name=epoch)
            self.epoch_metrics_df = pd.concat(
                [self.epoch_metrics_df, epoch_series.to_frame().T],
                axis=0
            )

            # --- Print main metrics for default threshold ---
            tm_acc  = self.get_epoch_metric('train_default_accuracy', epoch=epoch)
            vm_acc  = self.get_epoch_metric('val_default_accuracy', epoch=epoch)
            tm_tpr  = self.get_epoch_metric('train_default_TPR', epoch=epoch)
            vm_tpr  = self.get_epoch_metric('val_default_TPR', epoch=epoch)
            tm_rej  = self.get_epoch_metric('train_default_rej', epoch=epoch)
            vm_rej  = self.get_epoch_metric('val_default_rej', epoch=epoch)
            tm_roc_auc = self.get_epoch_metric('train_roc_auc', epoch=epoch)
            vm_roc_auc  = self.get_epoch_metric('val_roc_auc', epoch=epoch)

            print(f"Epoch {epoch} | default threshold:")
            print(f"  Train - Acc: {tm_acc:.4f}, Eff: {tm_tpr:.4f}, Rej: {tm_rej:.4f}, ROC AUC: {tm_roc_auc:.4f}")
            print(f"  Val   - Acc: {vm_acc:.4f}, Eff: {vm_tpr:.4f}, Rej: {vm_rej:.4f}, ROC AUC: {vm_roc_auc:.4f}")

            # --- Checkpoint saving ---
            if save_checkpoint:
                safe_epoch_frac = int(checkpoint_freq * epochs)
                if safe_epoch_frac == 0:
                    safe_epoch_frac = epochs + 1  # Avoid division by zero or zero modulo
                if epoch % safe_epoch_frac == 0 and epoch != 0:
                    print(f"Saving checkpoint at epoch {epoch}")
                    self.epoch_warmstart = epoch
                    file_path = f'{checkpoint_path}checkpoint_{epoch}.pt'
                    self.save_checkpoint(file_path)

            # --- Early stopping condition ---
            if patience_counter >= early_stopping_patience:
                self.last_epoch = epoch
                print(f"Early stopping at epoch {epoch}. Best epoch was {best_epoch} with val_loss={best_val_loss:.4f}")
                break

        # Restore model to best observed state after training loop finishes
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Model weights restored to best epoch {best_epoch}")


    def save_dataframe(self, file_name):
        """
        Save training and validation metrics collected during training into a CSV file.

        Args:
            file_name (str): Path to the output CSV file.

        Returns:
            pd.DataFrame: The saved DataFrame containing losses and accuracy/efficiency/rejection metrics.
        
        Notes:
            - Handles optional BCE loss metrics if enabled.
            - Converts any tensor metrics to float before saving.
        """

        # Prepare dictionary to save losses and metrics into a DataFrame
        data = {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
        }
        # Metrics related to accuracy, efficiency, and rejection (per class) for edge classification
        data["train_acc"] = [x.cpu().item() if torch.is_tensor(x) else float(x) for x in self.train_acc]
        data["val_acc"] = [x.cpu().item() if torch.is_tensor(x) else float(x) for x in self.val_acc]
        data["train_eff"] = [x.cpu().item() if torch.is_tensor(x) else float(x) for x in self.train_eff]
        data["val_eff"] = [x.cpu().item() if torch.is_tensor(x) else float(x) for x in self.val_eff]
        data["train_rej"] = [x.cpu().item() if torch.is_tensor(x) else float(x) for x in self.train_rej]
        data["val_rej"] = [x.cpu().item() if torch.is_tensor(x) else float(x) for x in self.val_rej]

        # Include BCE losses if the BCE flag is active
        if self.add_bce:
            data["ce_train_loss"] = self.ce_train_loss
            data["ce_val_loss"] = self.ce_val_loss
            # Commented out: node BCE losses are not saved currently
            # data["bce_nodes_train_loss"] = self.bce_nodes_train_loss
            # data["bce_nodes_val_loss"] = self.bce_nodes_val_loss
            data["bce_edges_train_loss"] = self.bce_edges_train_loss
            data["bce_edges_val_loss"] = self.bce_edges_val_loss

        # Create a pandas DataFrame and save it to CSV
        df = pd.DataFrame(data)
        df.to_csv(file_name)
        return df


    def compute_thresholds_and_metrics(self, y_true: np.ndarray, y_score: np.ndarray, loss, y_neutrals_id: np.ndarray, key_prefix: str, epoch=-1):
        """
        Compute performance metrics at multiple thresholds, both globally and for specific neutral particle types.

        Parameters:
        - y_true: true binary labels (0 or 1)
        - y_score: predicted scores (float)
        - loss: loss value for the epoch
        - y_neutrals_id: array of PDG IDs for neutral particles
        - key_prefix: prefix string to label metrics (e.g. 'train' or 'val')
        - epoch: current epoch number (default -1, unused)

        Returns:
        - A dictionary of metrics with metrics for the full dataset and per-particle subsamples.
        """
        metrics_dict = {}
        y_true = y_true.astype(int)

        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        # Remove first threshold (inf)
        thresholds = thresholds[1:]
        fpr = fpr[1:]
        tpr = tpr[1:]

        N_signal = int((y_true == 1).sum())
        N_background = int((y_true == 0).sum())

        S_arr = tpr * N_signal
        B_arr = fpr * N_background

        fom = np.divide(S_arr, np.sqrt(S_arr + B_arr), out=np.zeros_like(S_arr), where=(S_arr + B_arr) > 0)

        opt_idx = np.nanargmax(fom)
        opt_threshold = thresholds[opt_idx]
        tpr_at_opt = tpr[opt_idx]

        def find_threshold_for_tpr(target_tpr):
            idxs = np.where(tpr >= target_tpr)[0]
            return thresholds[idxs[0]] if idxs.size > 0 else thresholds[-1]

        tpr09_threshold = find_threshold_for_tpr(0.9)
        tpr099_threshold = find_threshold_for_tpr(0.99)

        threshold_info = [
            ('default', self.threshold),
            ('opt', opt_threshold),
            ('tpr0.9', tpr09_threshold),
            ('tpr0.99', tpr099_threshold),
        ]

        total_samples = y_true.shape[0]

        def compute_and_store_metrics(y_true_sub, y_score_sub, prefix):
            y_true_sub = y_true_sub.astype(int)
            total = y_true_sub.shape[0]
            for name, thr in threshold_info:
                y_pred_bin = (y_score_sub > thr).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_true_sub, y_pred_bin, labels=[0, 1]).ravel()

                tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                rej_val = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                prec_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                acc_val = (tp + tn) / total if total > 0 else 0.0
                bal_acc_val = 0.5 * (tpr_val + rej_val)

                key = f"{prefix}_{name}"
                metrics_dict[f"{key}_TP"] = int(tp)
                metrics_dict[f"{key}_FP"] = int(fp)
                metrics_dict[f"{key}_TN"] = int(tn)
                metrics_dict[f"{key}_FN"] = int(fn)
                metrics_dict[f"{key}_TPR"] = float(tpr_val)
                metrics_dict[f"{key}_rej"] = float(rej_val)
                metrics_dict[f"{key}_precision"] = float(prec_val)
                metrics_dict[f"{key}_accuracy"] = float(acc_val)
                metrics_dict[f"{key}_balanced_accuracy"] = float(bal_acc_val)

        # 1. Global metrics
        compute_and_store_metrics(y_true, y_score, key_prefix)

        # 2. Add global info
        metrics_dict[f"{key_prefix}_default_threshold_value"] = float(self.threshold)
        metrics_dict[f"{key_prefix}_opt_threshold_value"] = float(opt_threshold)
        metrics_dict[f"{key_prefix}_tpr0.9_threshold_value"] = float(tpr09_threshold)
        metrics_dict[f"{key_prefix}_tpr0.99_threshold_value"] = float(tpr099_threshold)
        metrics_dict[f"{key_prefix}_loss"] = float(loss)
        metrics_dict[f"{key_prefix}_roc_auc"] = float(roc_auc)

        self.tpr_and_threshold[key_prefix][epoch] = {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'threshold_default': self.threshold,
            'threshold_opt': opt_threshold,
            'tpr_at_opt': tpr_at_opt,
            'threshold_tpr_90': tpr09_threshold,
            'threshold_tpr_99': tpr099_threshold,
            'fom': fom
        }


        ### DEBUG TODO !!!!

        # '''

        # 3. Per-particle subsample metrics
        particle_masks = {
            22: "gamma",     # photon
            111: "pi0",      # neutral pion
            130: "k0L",      # K0_L
            310: "k0S",      # K0_S
            3122: "lambda0", # Lambda
        }

        known_ids = set(particle_masks.keys())
        masks = {}

        # Build masks for known particle types
        for pdg_id, id_suffix in particle_masks.items():
            mask = y_neutrals_id == pdg_id
            if np.sum(mask) > 0:
                self.particle_list.append(id_suffix)
                masks[id_suffix] = mask

        # Add mask for all other types
        mask_other = ~np.isin(y_neutrals_id, list(known_ids))
        if np.sum(mask_other) > 0:
            masks["other"] = mask_other

        # Compute metrics for each mask
        for id_suffix, mask in masks.items():
            y_true_sub = y_true[mask]
            y_score_sub = y_score[mask]
            sub_prefix = f"{key_prefix}_{id_suffix}"
            compute_and_store_metrics(y_true_sub, y_score_sub, sub_prefix)

        # '''

        return metrics_dict


    def save_metrics(self, file_name: str):
        """
        Save the DataFrame of epoch-wise metrics to a CSV file.

        Parameters:
        - file_name: path to the output CSV file

        Returns:
        - The DataFrame that was saved.
        """
        self.epoch_metrics_df.to_csv(file_name, index_label='epoch')
        return self.epoch_metrics_df


    def get_epoch_metric(self, column_name: str, epoch=None):
        """
        Retrieve metric values from the epoch metrics DataFrame.

        Parameters:
        - column_name: the metric column to retrieve
        - epoch: if None, returns the entire column as a NumPy array;
                if integer, returns the value for that epoch.

        Returns:
        - The metric values as a float or NumPy array.

        Raises:
        - KeyError if the column or epoch is not found.
        """
        if column_name not in self.epoch_metrics_df.columns:
            raise KeyError(f"Column '{column_name}' not found in epoch_metrics_df.")

        if epoch is None:
            # Return full column
            return self.epoch_metrics_df[column_name].values
        else:
            if epoch in self.epoch_metrics_df.index:
                return self.epoch_metrics_df.loc[epoch, column_name]
            else:
                raise KeyError(f"Epoch {epoch} not found in epoch_metrics_df.")
