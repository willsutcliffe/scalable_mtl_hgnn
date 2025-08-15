# Abstract base class module
from abc import ABC, abstractmethod

# Core libraries
import torch
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

# Plotting style for HEP
import mplhep
plt.style.use(mplhep.style.LHCb2)

# Metrics and uncertainty handling
from sklearn.metrics import roc_curve, auc
from uncertainties.unumpy import (
    uarray, 
    nominal_values as unp_n,
    std_devs as unp_s
)

# Custom utility functions
from wmpgnn.util.functions import (
    NOW, plt_pull, ks_test, centers, hist,
    plt_smooth, batched_predict_proba
)

class NeutralsTrainer(ABC):
    """
    Abstract base class for model training loops.

    Manages:
      - Configuration, model, and data loaders
      - Epoch‐wise logging of loss and per‐class accuracies
      - Checkpoint saving
      - Simple plotting utilities for loss and accuracy

    Subclasses must implement:
      - eval_one_epoch(self, train: bool) -> (float, Tensor)
      - train(self, epochs: int, learning_rate: float) -> None
    """

    def __init__(self, config, model, train_loader, val_loader):
        """
        Initialize core training attributes.

        Args:
            config (dict):
                Hyperparameter and environment settings.
            model (torch.nn.Module):
                The PyTorch model to be trained.
            train_loader (DataLoader):
                Yields training batches.
            val_loader (DataLoader):
                Yields validation batches.

        Attributes:
            config (dict): As passed in.
            model (torch.nn.Module): As passed in.
            train_loader, val_loader: Data loaders.
            train_acc, val_acc, ... (List[Tensor]): Per‐epoch class accuracies and other metrics
            train_loss, val_loss (List[float]): Per‐epoch loss values.
            epochs (List[int]): Epoch indices logged.
        """
        # Configuration and model
        self.config = config
        self.model = model

        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Metric tracking
        self.train_acc = []
        self.val_acc = []
        self.train_eff = []
        self.val_eff = []
        self.train_rej = []
        self.val_rej = []

        # Associated errors
        self.train_acc_err = []
        self.val_acc_err = []
        self.train_eff_err = []
        self.val_eff_err = []
        self.train_rej_err = []
        self.val_rej_err = []

        # Loss tracking
        self.train_loss = []
        self.val_loss = []

        # Epoch tracker
        self.epochs = []

        # Number of classes for the neutrals (from config)
        self.neutrals_classes = config.get('model.neutrals_classes')

        # Start epoch index
        self.epoch_warmstart = 0

        # Prediction and label buffers
        self.train_predictions = []
        self.train_labels = []
        self.train_neutrals_id = []
        self.val_predictions = []
        self.val_labels = []
        self.val_neutrals_id = []

        # DataFrame to log metrics per epoch
        self.epoch_metrics_df = pd.DataFrame()

        # Dictionary to store TPR vs threshold curves
        self.tpr_and_threshold = {'train': {}, 'val': {}}

        self.particle_list = []

    @abstractmethod
    def eval_one_epoch(self, train=True):
        """
        Run a single epoch of evaluation (and training if train=True).

        Must:
          - Iterate over the appropriate loader.
          - Compute batch‐wise loss and per‐class accuracy.

        Args:
            train (bool):
                If True, perform backprop through train_loader;
                if False, evaluate on val_loader.
        """
        pass

    @abstractmethod
    def train(self, epochs=10, learning_rate=0.001):
        """
        Execute the full training loop.

        Must:
          - Loop over `epochs`
          - Call `eval_one_epoch(train=True)` and `eval_one_epoch(train=False)`
          - Append losses/accuracies to internal logs
          - Optionally print progress

        Args:
            epochs (int): Number of epochs to train.
            learning_rate (float): Learning rate for optimizer.
        """
        pass

    def save_model(self, file_name, save_config=False):
        """
        Save the model to disk. Optionally save the config as a text file.
        """
        # Save model checkpoint (state dict, epoch, metrics)
        self.save_checkpoint(file_path=file_name)

        # Save configuration file as human-readable txt if requested
        if save_config:
            print("Saving config file as txt file")
            self.config.print(file_name.replace('.pt', '.txt'))
            with open(file_name.replace('.pt', '.txt'), 'a') as f:
                f.write(f"Date: {NOW(fmt='%Y-%m-%d %H:%M:%S')}\n")

    @abstractmethod
    def save_checkpoint(self, epoch: int, train_metrics: dict, val_metrics: dict, file_path: str):
        """
        Abstract method to save a full training checkpoint.
        """
        pass

    @abstractmethod
    def load_checkpoint(self, file_path=None):
        """
        Abstract method to load a previously saved checkpoint.
        """
        pass

    def get_history(self):
        """
        Return a dictionary containing the full training and validation history.
        """
        history = {
            'train_loss': self.train_loss,
            'train_acc': self.train_acc,
            'train_eff': self.train_eff,
            'train_rej': self.train_rej,
            'train_acc_err': self.train_acc_err,
            'train_eff_err': self.train_eff_err,
            'train_rej_err': self.train_rej_err,
            'val_loss': self.val_loss,
            'val_acc': self.val_acc,
            'val_eff': self.val_eff,
            'val_rej': self.val_rej,
            'val_acc_err': self.val_acc_err,
            'val_eff_err': self.val_eff_err,
            'val_rej_err': self.val_rej_err,
            'ce_train_loss': self.ce_train_loss,
            'ce_val_loss': self.ce_val_loss,
            # 'bce_nodes_train_loss': self.bce_nodes_train_loss,
            # 'bce_nodes_val_loss': self.bce_nodes_val_loss,
            'bce_edges_train_loss': self.bce_edges_train_loss,
            'bce_edges_val_loss': self.bce_edges_val_loss,
        }
        return history

    def set_history(self, history):
        """
        Restore the training and validation history from a dictionary.
        """
        self.train_loss    = history['train_loss']
        self.train_acc     = history['train_acc']
        self.train_eff     = history['train_eff']
        self.train_rej     = history['train_rej']
        self.train_acc_err = history['train_acc_err']
        self.train_eff_err = history['train_eff_err']
        self.train_rej_err = history['train_rej_err']
        self.val_loss      = history['val_loss']
        self.val_acc       = history['val_acc']
        self.val_eff       = history['val_eff']
        self.val_rej       = history['val_rej']
        self.val_acc_err   = history['val_acc_err']
        self.val_eff_err   = history['val_eff_err']
        self.val_rej_err   = history['val_rej_err']
        self.ce_train_loss        = history['ce_train_loss']
        self.ce_val_loss          = history['ce_val_loss']
        # self.bce_nodes_train_loss = history['bce_nodes_train_loss']
        # self.bce_nodes_val_loss   = history['bce_nodes_val_loss']
        self.bce_edges_train_loss = history['bce_edges_train_loss']
        self.bce_edges_val_loss   = history['bce_edges_val_loss']

    def save_dataframe(self, file_name):
        """
        Placeholder for saving epoch_metrics_df to a file.
        Implement in subclass if needed.
        """
        pass


    def plot_loss(self, file_name="loss.png", show=True):
        # Plot training and validation loss over epochs
        plt.plot(self.train_loss, label="Train Loss")
        plt.plot(self.val_loss, label="Validation Loss")

        plt.xlabel('Epoch')
        plt.ylabel('Binary Cross Entropy Loss')
        plt.grid()
        plt.legend()

        if show:
            plt.show()

        plt.savefig(file_name)

    def plot_predictions(self, path, file_name="pred.png", epoch=-1, show=True):
        """
        Plot prediction distributions for train and validation samples,
        with vertical lines for various thresholds (default, opt, tpr0.9, tpr0.99).
        """

        def to_numpy(tensor):
            # Convert torch.Tensor to numpy array
            if isinstance(tensor, torch.Tensor):
                return tensor.detach().cpu().numpy()
            return tensor

        # 1) Build the response dictionary: (predictions, weights)
        data = {
            'Bkg (train)': (
                to_numpy(self.train_predictions[epoch][self.train_labels[epoch] == 0]),
                to_numpy(torch.ones_like(self.train_labels[epoch][self.train_labels[epoch] == 0])),
            ),
            'Bkg (val)': (
                to_numpy(self.val_predictions[epoch][self.val_labels[epoch] == 0]),
                to_numpy(torch.ones_like(self.val_labels[epoch][self.val_labels[epoch] == 0])),
            ),
            'Signal (train)': (
                to_numpy(self.train_predictions[epoch][self.train_labels[epoch] == 1]),
                to_numpy(torch.ones_like(self.train_labels[epoch][self.train_labels[epoch] == 1])),
            ),
            'Signal (val)': (
                to_numpy(self.val_predictions[epoch][self.val_labels[epoch] == 1]),
                to_numpy(torch.ones_like(self.val_labels[epoch][self.val_labels[epoch] == 1])),
            ),
        }
        responses = data

        # Define binning and compute histogram bin centers with uncertainties
        bins = np.linspace(0, 1, 30)
        x, xerr = centers(bins, xerr=True)

        # Create subplots: main plot + 2 pull plots
        figure, (ax, pull1, pull2) = plt.subplots(
            3, sharex=True, figsize=(16, 10),
            gridspec_kw=dict(height_ratios=(4, 1, 1), hspace=0),
        )

        # 2) Compute histograms with uncertainties, normalized to sum 1
        hists = {
            key: uarray(h, he) / np.sum(h)
            for key, (response, weight) in responses.items()
            for (_, h, he) in (hist(response, weight, bins=bins),)
        }

        # Compute pulls: val - train
        signal = hists['Signal (val)'] - hists['Signal (train)']
        bkg    = hists['Bkg (val)']    - hists['Bkg (train)']

        # Determine y-axis limits based on min/max of histogram content
        ymin = min(
            (
                np.min(np.fmax((.5 * unp_n(h))[unp_n(h) > 0],
                            (unp_n(h) - unp_s(h))[unp_n(h) > 0]))
            )
            for h in hists.values()
        )
        ymax = max(unp_n(h).max() + unp_s(h).max() for h in hists.values()) * 1.1

        ax.set_ylim(bottom=ymin, top=ymax)

        # 3) Plot histograms: smooth for train, error bars for val
        for key, h in hists.items():
            color = (.1, .1, .8) if 'Signal' in key else (.8, .1, .1)
            if 'train' in key:
                plt_smooth(
                    ax, x, unp_n(h), unp_s(h),
                    label=key, color=(*color, .7)
                )
            else:
                ax.errorbar(
                    x, unp_n(h), unp_s(h), xerr=xerr, label=key,
                    linestyle='', marker='o', markersize=.5, linewidth=.3,
                    markeredgewidth=.3, capsize=.5, color=color
                )

        # 4) Plot pulls (normalized residuals between val and train)
        plt_pull(pull1, bins, unp_n(signal), 0, err=unp_s(signal))
        plt_pull(pull2, bins, unp_n(bkg),    0, err=unp_s(bkg))

        pull1.set_ylabel(
            'signal\n' r'$\frac{\mathrm{val} - \mathrm{train}}{\sigma}$',
            loc='center'
        )
        pull2.set_ylabel(
            'bkg\n' r'$\frac{\mathrm{val} - \mathrm{train}}{\sigma}$',
            loc='center'
        )

        # 5) Plot vertical threshold lines at current epoch
        last_epoch = epoch
        thresholds = ['default', 'opt', 'tpr0.9', 'tpr0.99']
        colors = ['tab:cyan', 'tab:orange', 'tab:green', 'tab:purple']

        for idx, th in enumerate(thresholds):
            if th == 'default':
                thr_val = self.threshold
            else:
                thr_val = self.get_epoch_metric(f"val_{th}_threshold_value", epoch=last_epoch)

            color = colors[idx]
            ax.axvline(
                thr_val,
                linestyle='--',
                color=color,
                label=f"{th.title()} (th={thr_val:.2f})"
            )

        # 6) Adjust tick positions to avoid overlap
        for label in pull1.get_yticklabels():
            label.set_y(label.get_position()[1] + 1)

        for label in pull2.get_yticklabels():
            label.set_y(label.get_position()[1] - 1)

        for label in ax.get_xticklabels():
            label.set_x(label.get_position()[0] + 1)

        # Final formatting
        plt.xlim(bins[0], bins[-1])
        plt.xlabel('Predictions')
        ax.set_title(f'Predictions Distribution at Epoch {epoch}')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=True,
                facecolor='white', edgecolor='black', framealpha=1)

        plt.tight_layout()
        plt.subplots_adjust(right=0.75)

        if show:
            plt.show()

        # 7) Save the final figure
        output_path = os.path.join(path, "predictions_distribution", file_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        figure.savefig(output_path)

    def plot_roc_auc(self, path, file_name=None, epoch=-1, show=True):
        """
        Plot ROC curves for train and validation at a given epoch.
        Annotate points and vertical/horizontal lines at the 4 thresholds stored in epoch_metrics_df.
        """
        if file_name is None:
            file_name = f"roc_epoch_{epoch}.png"

        # 1) Retrieve predictions and labels for train and validation
        y_train_pred = self.train_predictions[epoch]
        y_train_true = self.train_labels[epoch]
        y_val_pred   = self.val_predictions[epoch]
        y_val_true   = self.val_labels[epoch]

        # Convert torch tensors to NumPy arrays if needed
        def to_numpy(x):
            if hasattr(x, "detach"):
                return x.detach().cpu().numpy()
            return x

        y_train_pred = to_numpy(y_train_pred)
        y_train_true = to_numpy(y_train_true)
        y_val_pred   = to_numpy(y_val_pred)
        y_val_true   = to_numpy(y_val_true)

        # 2) Compute ROC curves and AUCs
        fpr_train, tpr_train, _ = roc_curve(y_train_true, y_train_pred)
        roc_auc_train = auc(fpr_train, tpr_train)

        fpr_val, tpr_val, thresholds_val = roc_curve(y_val_true, y_val_pred)
        roc_auc_val = auc(fpr_val, tpr_val)

        # 3) Plot ROC curves
        plt.figure(figsize=(14, 8))
        plt.plot(fpr_train, tpr_train, linestyle="--", linewidth=2, color="blue",
                label=f"Train ROC (AUC = {roc_auc_train:.3f})")
        plt.plot(fpr_val, tpr_val, linestyle="-", linewidth=3, color="red",
                label=f"Val ROC   (AUC = {roc_auc_val:.3f})")

        # 4) Draw threshold markers (vertical/horizontal lines + scatter points)
        thresholds = ["default", "opt", "tpr0.9", "tpr0.99"]
        colors = ["tab:cyan", "tab:orange", "tab:green", "tab:purple"]
        last_epoch = epoch

        for idx, th in enumerate(thresholds):
            # Get threshold value
            thr_val = self.threshold if th == "default" else self.get_epoch_metric(
                f"val_{th}_threshold_value", epoch=last_epoch)

            # Find closest threshold on the ROC curve
            idx_closest = np.argmin(np.abs(thresholds_val - thr_val))
            fpr_pt = fpr_val[idx_closest]
            tpr_pt = tpr_val[idx_closest]

            color = colors[idx]
            # Draw vertical/horizontal lines
            plt.axvline(fpr_pt, linestyle='--', color=color)
            plt.axhline(tpr_pt, linestyle='--', color=color)

            # Draw point with label
            label = f"{th.title()} (th={thr_val:.2f})"
            plt.scatter(fpr_pt, tpr_pt, color=color, s=50, zorder=5, label=label)

        # 5) Final plot styling
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curves at Epoch {epoch}")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1),
                frameon=True, facecolor='white', edgecolor='black', framealpha=1)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)
        plt.tight_layout()
        plt.subplots_adjust(right=0.7)

        # 6) Save or display
        output_path = os.path.join(path, "roc_auc", file_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)

        if show:
            plt.show()
        else:
            plt.close()

    def plot_accuracy(self, particle, file_name="acc.png", show=True):
        """
        Plot train/val accuracy for the four thresholds, with threshold values in the labels.
        """
        epochs_list = self.epoch_metrics_df.index.values
        thresholds = ['default', 'opt', 'tpr0.9', 'tpr0.99']
        colors = ['tab:cyan', 'tab:orange', 'tab:green', 'tab:purple']
        markers = {'train': 'o', 'val': 's'}
        last_epoch = self.epoch_metrics_df.index.max()

        fig, ax = plt.subplots(figsize=(14, 8))

        for idx, th in enumerate(thresholds):
            # Retrieve accuracy values for both train and val
            train_vals = self.get_epoch_metric(f"train_{particle}{th}_accuracy", epoch=None)
            val_vals   = self.get_epoch_metric(f"val_{particle}{th}_accuracy", epoch=None)

            # Get corresponding threshold values
            if th == 'default':
                th_value_train = th_value_val = self.threshold
            else:
                th_value_train = self.get_epoch_metric(f"train_{th}_threshold_value", epoch=last_epoch)
                th_value_val   = self.get_epoch_metric(f"val_{th}_threshold_value",   epoch=last_epoch)
            # Format labels
            label_train = f"Train {th.title()} (th={th_value_train:.2f})"
            label_val   = f"Val   {th.title()} (th={th_value_val:.2f})"

            # Plot curves
            ax.plot(epochs_list, train_vals, marker=markers['train'], markersize=4,
                    color=colors[idx], label=label_train)
            ax.plot(epochs_list, val_vals, marker=markers['val'], linestyle='--',
                    color=colors[idx], label=label_val)

        # Styling
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Train and Validation (4 thresholds)")
        ax.grid(True)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1),
                frameon=True, facecolor='white', edgecolor='black', framealpha=1)
        fig.tight_layout()
        fig.subplots_adjust(right=0.7)

        if show:
            plt.show()
        fig.savefig(file_name)



    def plot_efficiency(self, particle,file_name="eff.png", show=True):
        """
        Plot train/val efficiency (True Positive Rate) for the four thresholds, with threshold values in the labels.
        """
        epochs_list = self.epoch_metrics_df.index.values
        thresholds = ['default', 'opt', 'tpr0.9', 'tpr0.99']
        colors = ['tab:cyan', 'tab:orange', 'tab:green', 'tab:purple']
        markers = {'train': 'o', 'val': 's'}
        last_epoch = self.epoch_metrics_df.index.max()

        fig, ax = plt.subplots(figsize=(14, 8))

        for idx, th in enumerate(thresholds):
            train_vals = self.get_epoch_metric(f"train_{particle}{th}_TPR", epoch=None)
            val_vals   = self.get_epoch_metric(f"val_{particle}{th}_TPR", epoch=None)

            if th == 'default':
                th_value_train = th_value_val = self.threshold
            else:
                th_value_train = self.get_epoch_metric(f"train_{th}_threshold_value", epoch=last_epoch)
                th_value_val   = self.get_epoch_metric(f"val_{th}_threshold_value",   epoch=last_epoch)

            label_train = f"Train {th.title()} (th={th_value_train:.2f})"
            label_val   = f"Val   {th.title()} (th={th_value_val:.2f})"

            ax.plot(epochs_list, train_vals, marker=markers['train'], markersize=4,
                    color=colors[idx], label=label_train)
            ax.plot(epochs_list, val_vals, marker=markers['val'], linestyle='--',
                    color=colors[idx], label=label_val)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Efficiency (TPR)")
        ax.set_title("Train and Validation (4 thresholds)")
        ax.grid(True)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1),
                frameon=True, facecolor='white', edgecolor='black', framealpha=1)
        fig.tight_layout()
        fig.subplots_adjust(right=0.7)

        if show:
            plt.show()
        fig.savefig(file_name)


    def plot_rejection(self, particle, file_name="rej.png", show=True):
        """
        Plot train/val rejection pour les quatre thresholds, avec la valeur du seuil.
        """
        epochs_list = self.epoch_metrics_df.index.values
        thresholds = ['default', 'opt', 'tpr0.9', 'tpr0.99']
        colors = ['tab:cyan', 'tab:orange', 'tab:green', 'tab:purple']
        markers = {'train': 'o', 'val': 's'}

        last_epoch = self.epoch_metrics_df.index.max()

        fig, ax = plt.subplots(figsize=(14, 8))

        for idx, th in enumerate(thresholds):
            train_col = f"train_{particle}{th}_rej"
            val_col   = f"val_{particle}{th}_rej"

            train_vals = self.get_epoch_metric(train_col, epoch=None)
            val_vals   = self.get_epoch_metric(val_col, epoch=None)

            if th == 'default':
                th_value_train = th_value_val = self.threshold
            else:
                th_value_train = self.get_epoch_metric(f"train_{th}_threshold_value", epoch=last_epoch)
                th_value_val   = self.get_epoch_metric(f"val_{th}_threshold_value",   epoch=last_epoch)

            label_train = f"Train {th.title()} (th={th_value_train:.2f})"
            label_val   = f"Val   {th.title()} (th={th_value_val:.2f})"

            ax.plot(epochs_list, train_vals,
                    marker=markers['train'],
                    markersize=4, 
                    color=colors[idx],
                    label=label_train)
            ax.plot(epochs_list, val_vals,
                    marker=markers['val'],
                    linestyle='--',
                    color=colors[idx],
                    label=label_val)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Rejection")
        ax.set_title("Train and Validation (4 thresholds)")
        ax.grid(True)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=True, facecolor='white', edgecolor='black', framealpha=1)
        fig.tight_layout()
        fig.subplots_adjust(right=0.7)
        if show:
            plt.show()
        fig.savefig(file_name)

    def plot_precision(self, particle, file_name="prec.png", show=True):
        """
        Plot train/val precision pour les quatre thresholds, avec la valeur du seuil.
        """
        epochs_list = self.epoch_metrics_df.index.values
        thresholds = ['default', 'opt', 'tpr0.9', 'tpr0.99']
        colors = ['tab:cyan', 'tab:orange', 'tab:green', 'tab:purple']
        markers = {'train': 'o', 'val': 's'}

        last_epoch = self.epoch_metrics_df.index.max()

        fig, ax = plt.subplots(figsize=(14, 8))

        for idx, th in enumerate(thresholds):
            train_col = f"train_{particle}{th}_precision"
            val_col   = f"val_{particle}{th}_precision"

            train_vals = self.get_epoch_metric(train_col, epoch=None)
            val_vals   = self.get_epoch_metric(val_col, epoch=None)

            if th == 'default':
                th_value_train = th_value_val = self.threshold
            else:
                th_value_train = self.get_epoch_metric(f"train_{th}_threshold_value", epoch=last_epoch)
                th_value_val   = self.get_epoch_metric(f"val_{th}_threshold_value",   epoch=last_epoch)

            label_train = f"Train {th.title()} (th={th_value_train:.2f})"
            label_val   = f"Val   {th.title()} (th={th_value_val:.2f})"

            ax.plot(epochs_list, train_vals,
                    marker=markers['train'],
                    markersize=4, 
                    color=colors[idx],
                    label=label_train)
            ax.plot(epochs_list, val_vals,
                    marker=markers['val'],
                    linestyle='--',
                    color=colors[idx],
                    label=label_val)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Precision")
        ax.set_title("Train and Validation (4 thresholds)")
        ax.grid(True)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=True, facecolor='white', edgecolor='black', framealpha=1)
        fig.tight_layout()
        fig.subplots_adjust(right=0.7)
        if show:
            plt.show()
        fig.savefig(file_name)

    def plot_balanced_accuracy(self, particle, file_name="bal_acc.png", show=True):
        """
        Plot train/val balanced accuracy pour les quatre thresholds, avec la valeur du seuil.
        """
        epochs_list = self.epoch_metrics_df.index.values
        thresholds = ['default', 'opt', 'tpr0.9', 'tpr0.99']
        colors = ['tab:cyan', 'tab:orange', 'tab:green', 'tab:purple']
        markers = {'train': 'o', 'val': 's'}

        last_epoch = self.epoch_metrics_df.index.max()

        fig, ax = plt.subplots(figsize=(14, 8))

        for idx, th in enumerate(thresholds):
            train_col = f"train_{particle}{th}_balanced_accuracy"
            val_col   = f"val_{particle}{th}_balanced_accuracy"

            train_vals = self.get_epoch_metric(train_col, epoch=None)
            val_vals   = self.get_epoch_metric(val_col, epoch=None)

            if th == 'default':
                th_value_train = th_value_val = self.threshold
            else:
                th_value_train = self.get_epoch_metric(f"train_{th}_threshold_value", epoch=last_epoch)
                th_value_val   = self.get_epoch_metric(f"val_{th}_threshold_value", epoch=  last_epoch)

            label_train = f"Train {th.title()} (th={th_value_train:.2f})"
            label_val   = f"Val   {th.title()} (th={th_value_val:.2f})"

            ax.plot(epochs_list, train_vals,
                    marker=markers['train'],
                    markersize=4, 
                    color=colors[idx],
                    label=label_train)
            ax.plot(epochs_list, val_vals,
                    marker=markers['val'],
                    linestyle='--',
                    color=colors[idx],
                    label=label_val)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Balanced Accuracy")
        ax.set_title("Train and Validation (4 thresholds)")
        ax.grid(True)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=True, facecolor='white', edgecolor='black', framealpha=1)
        fig.tight_layout()
        fig.subplots_adjust(right=0.7)
        if show:
            plt.show()
        fig.savefig(file_name)

    def plot_tpr_thresholds(self, path, file_name, key_prefix: str = 'val', epoch: int =-1, show: bool = True):
        """
        Plot TPR vs. threshold for 'train' or 'val' (given key_prefix) 
        and mark the four cutoffs (default, opt, tpr=0.9, tpr=0.99) 
        with vertical lines and points. The style (colors, legend format) 
        matches that of plot_efficiency.
        """
        if key_prefix not in self.tpr_and_threshold:
            raise ValueError(f"No stored data for key_prefix='{key_prefix}'. Have you already called compute_thresholds_and_metrics for '{key_prefix}'?")
        if epoch not in self.tpr_and_threshold[key_prefix]:
            raise ValueError(f"No stored TPR‐vs‐threshold data for '{key_prefix}' at epoch={epoch}.")

        data = self.tpr_and_threshold[key_prefix][epoch]
        thresholds = data['thresholds']
        tpr_curve = data['tpr']

        # Retrieve all four thresholds
        th_default = data['threshold_default']
        th_opt = data['threshold_opt']
        th_tpr90 = data['threshold_tpr_90']
        th_tpr99 = data['threshold_tpr_99']
        tpr_at_opt = data['tpr_at_opt']

        # Define a consistent color scheme (same order as plot_efficiency: default, opt, tpr0.9, tpr0.99)
        colors = {
            'default': 'tab:cyan',
            'opt': 'tab:orange',
            'tpr0.9': 'tab:green',
            'tpr0.99': 'tab:purple'
        }

        plt.figure(figsize=(12, 10))
        # Plot the ROC‐derived TPR curve (TPR vs. threshold)
        plt.plot(thresholds, tpr_curve,
                 marker="o", linestyle="-", markersize=3,
                 color='red', label="TPR Curve")

        # For each of the four thresholds, draw a vertical line and a marker point:
        # 1) default threshold
        #    We draw a vertical line at th_default. The point on the TPR curve at that threshold 
        #    might not correspond exactly to the curve arrays—so we can approximate by computing
        #    the TPR at that cutoff defaultly if needed. But typically, we just draw the vertical line
        #    at x = th_default and let the legend note its value.
        plt.axvline(th_default, color=colors['default'], linestyle="--",
                    label=f"default (th={th_default:.3f})")

        # 2) Optimal threshold
        plt.axvline(th_opt, color=colors['opt'], linestyle="--",
                    label=f"Optimal (th={th_opt:.3f}, TPR={tpr_at_opt:.3f})")

        # 3) TPR = 0.9 threshold
        #    We display a horizontal marker at y=0.9 and drop a vertical
        #    line at th_tpr90. To show the marker on the curve, we can scatter at (th_tpr90, 0.9).
        plt.scatter([th_tpr90], [0.9], color=colors['tpr0.9'], zorder=3)
        plt.axvline(th_tpr90, color=colors['tpr0.9'], linestyle="--",
                    label=f"TPR=0.90 (th={th_tpr90:.3f})")

        # 4) TPR = 0.99 threshold
        plt.scatter([th_tpr99], [0.99], color=colors['tpr0.99'], zorder=3)
        plt.axvline(th_tpr99, color=colors['tpr0.99'], linestyle="--",
                    label=f"TPR=0.99 (th={th_tpr99:.3f})")

        plt.xlabel("Threshold")
        plt.ylabel("True Positive Rate (TPR)")
        plt.title(f"TPR vs. Threshold ({key_prefix.title()}, Epoch {epoch})")
        plt.legend(loc='best', frameon=True, facecolor='white', edgecolor='black', framealpha=1)
        plt.grid(True)

        if show:
            plt.show()
        plt.tight_layout()
        output_path = os.path.join(path, "thresholds", file_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()

    def plot_fom_vs_threshold(self, path, file_name, key_prefix: str = 'val', epoch: int = -1, show: bool = True):
        """
        Plot the Figure of Merit (FOM = S/√(S+B)) vs. threshold for 'train' or 'val' 
        (given key_prefix). Mark the optimal threshold with a vertical line.
        """
        if key_prefix not in self.tpr_and_threshold:
            raise ValueError(f"No stored data for key_prefix='{key_prefix}'. Have you already called compute_thresholds_and_metrics for '{key_prefix}'?")
        if epoch not in self.tpr_and_threshold[key_prefix]:
            raise ValueError(f"No stored FOM‐vs‐threshold data for '{key_prefix}' at epoch={epoch}.")

        data = self.tpr_and_threshold[key_prefix][epoch]
        thresholds = data['thresholds']
        fom_curve = data['fom']
        th_opt = data['threshold_opt']

        plt.figure(figsize=(10, 8))
        plt.plot(thresholds, fom_curve,
                 marker="o", linestyle="-", markersize=3,
                 label="FOM = S / √(S + B)")

        # Vertical line at optimal threshold
        plt.axvline(th_opt, color='orange', linestyle="--",
                    label=f"Optimal (th={th_opt:.3f})")

        plt.xlabel("Threshold")
        plt.ylabel("FOM")
        plt.title(f"FOM vs. Threshold ({key_prefix.title()}, Epoch {epoch})")
        plt.legend(loc='best', frameon=True, facecolor='white', edgecolor='black', framealpha=1)
        plt.grid(True)

        if show:
            plt.show()
        plt.tight_layout()

        output_path = os.path.join(path, "fom", file_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)

        plt.close()

   