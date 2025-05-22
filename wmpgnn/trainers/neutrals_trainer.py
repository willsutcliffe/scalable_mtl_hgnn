from abc import ABC, abstractmethod
import torch
import matplotlib.pyplot as plt
import os
import numpy as np
import mplhep
from sklearn.metrics import roc_curve, auc
from uncertainties.unumpy import (uarray, nominal_values as unp_n,
                                  std_devs as unp_s)
from wmpgnn.util.functions import NOW, plt_pull, ks_test, centers, hist, plt_smooth, batched_predict_proba
# plt.style.use(mplhep.style.LHCb2)

class NeutralsTrainer(ABC):

    def __init__(self, config, model, train_loader, val_loader):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_acc = []
        self.val_acc = []
        self.train_eff = []
        self.val_eff = []
        self.train_rej = []
        self.val_rej = []
        self.train_acc_err = []
        self.val_acc_err = []
        self.train_eff_err = []
        self.val_eff_err = []
        self.train_rej_err = []
        self.val_rej_err = []
        self.train_loss = []
        self.val_loss = []
        self.epochs = []
        self.neutrals_classes = config.get('model.neutrals_classes')
        self.epoch_warmstart = 0
        self.train_predictions = []
        self.train_labels = []
        self.val_predictions = []
        self.val_labels = []

    @abstractmethod
    def eval_one_epoch(self, train=True):
        pass

    @abstractmethod
    def train(self, epochs=10, learning_rate=0.001):
        pass

    def save_model(self, file_name, save_config=False):
        #torch.save(self.model.state_dict(), file_name)
        self.save_checkpoint(file_path=file_name)
        if save_config: # print config file
            print("Saving config file as txt file")
            self.config.print(file_name.replace('.pt','.txt'))
            # append date time
            with open(file_name.replace('.pt','.txt'), 'a') as f:
                f.write(f"Date: {NOW(fmt='%Y-%m-%d %H:%M:%S')}\n")
    
    @abstractmethod
    def save_checkpoint(self, epoch:int, train_metrics:dict, val_metrics:dict, file_path:str):
        pass
    
    @abstractmethod
    def load_checkpoint(self, file_path=None):
        pass

    def get_history(self):
        """Returns the training and validation history of the model's metrics"""
        history = {}
        history['train_loss']    = self.train_loss
        history['train_acc']     = self.train_acc
        history['train_eff']     = self.train_eff
        history['train_rej']     = self.train_rej
        history['train_acc_err'] = self.train_acc_err
        history['train_eff_err'] = self.train_eff_err
        history['train_rej_err'] = self.train_rej_err
        history['val_loss']      = self.val_loss
        history['val_acc']       = self.val_acc
        history['val_eff']       = self.val_eff
        history['val_rej']       = self.val_rej
        history['val_acc_err']   = self.val_acc_err
        history['val_eff_err']   = self.val_eff_err
        history['val_rej_err']   = self.val_rej_err
        history['ce_train_loss']        = self.ce_train_loss
        history['ce_val_loss']          = self.ce_val_loss
        # history['bce_nodes_train_loss'] = self.bce_nodes_train_loss
        # history['bce_nodes_val_loss']   = self.bce_nodes_val_loss
        history['bce_edges_train_loss'] = self.bce_edges_train_loss
        history['bce_edges_val_loss']   = self.bce_edges_val_loss
        return history
    
    def set_history(self, history):
        """set the training and validation history of the model's metrics"""
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
        pass

    def plot_loss(self, file_name="loss.png", show=True):
        

        plt.plot(self.train_loss, label="Train Loss")
        plt.plot(self.val_loss, label="Validation Loss")

        plt.xlabel('epoch')
        plt.ylabel('Cross Entropy Loss')
        plt.grid()

        plt.legend()
        if show:
            plt.show()
        plt.savefig(file_name)

    def plot_predictions(self, path, file_name="pred.png", epoch=-1, show=True):
        """Plot the predicitions for the training and validations samples"""

        def to_numpy(tensor):
            if isinstance(tensor, torch.Tensor):
                # Detach the tensor from the computation graph before converting to NumPy
                return tensor.detach().cpu().numpy()
            return tensor

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
        # responses = {
        #     key: (batched_predict_proba(model, sample)[:, 1], weight)
        #     for key, (sample, weight) in data.items()
        # }
        responses = data
        fontsize = 20
        bins = np.linspace(0, 1, 30)
        x, xerr = centers(bins, xerr=True)
        figure, (ax, pull1, pull2) = plt.subplots(
                3, sharex=True, figsize=(12, 10),
                gridspec_kw=dict(height_ratios=(4, 1, 1), hspace=0),
        )
        hists = {key: uarray(h, he) / np.sum(h)
                for key, (response, weight) in responses.items()
                for (_, h, he) in (hist(response, weight, bins=bins), )}
        ylim = min((np.min(np.fmax((.5 * unp_n(h))[unp_n(h) > 0],
                                (unp_n(h) - unp_s(h))[unp_n(h) > 0])))
                for h in hists.values())
        for key, h in hists.items():
            color = ((.1, .1, .8) if 'Signal' in key else (.8, .1, .1))
            if 'train' in key:
                plt_smooth(ax, x, unp_n(h), unp_s(h), label=key, color=(*color, .7))
            else:
                ax.errorbar(x, unp_n(h), unp_s(h), xerr=xerr, label=key,
                            linestyle='', marker='o', markersize=.5, linewidth=.3,
                            markeredgewidth=.3, capsize=.5, color=color)
        signal = hists['Signal (val)'] - hists['Signal (train)']
        bkg = hists['Bkg (val)'] - hists['Bkg (train)']
        plt_pull(pull1, bins, unp_n(signal), 0, err=unp_s(signal))
        plt_pull(pull2, bins, unp_n(bkg), 0, err=unp_s(bkg))
        pull1.set_ylabel('signal\n'
                        r'$\frac{\mathrm{val} - \mathrm{train}}{\sigma}$',
                        loc='center', fontsize=fontsize)
        pull2.set_ylabel('bkg\n'
                        r'$\frac{\mathrm{val} - \mathrm{train}}{\sigma}$',
                        loc='center', fontsize=fontsize)
        ax.set_ylim(bottom=ylim)
        ax.tick_params(axis='both', labelsize=fontsize)        # Change 16 par ce que tu veux
        pull1.tick_params(axis='both', labelsize=fontsize)     # Pour les subplots des pulls
        pull2.tick_params(axis='both', labelsize=fontsize)
        # ax.set_yscale('log')
        plt.xlim(bins[0], bins[-1])
        plt.xlabel('Predicitions', fontsize=fontsize)
        ax.legend(loc='best', fontsize=fontsize)
        # ax.set_title(ks_test(responses), size='medium')
        ax.set_title(f'Predictions distribution at Epoch {epoch}', fontsize=fontsize+2)
        if show:
            plt.show()
        output_path = os.path.join(path, "predictions_distribution", file_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        figure.savefig(output_path)

    def plot_roc_auc(self, path, file_name=None, epoch=-1, show=True):
        """
        Plot ROC AUC curves for training and validation splits at a given epoch.
        """
        # Set default filename if not provided
        if file_name is None:
            file_name = f"roc_epoch_{epoch}.png"

        # Extract predictions and labels for train and validation
        y_train_pred = self.train_predictions[epoch]
        y_train_true = self.train_labels[epoch]
        y_val_pred = self.val_predictions[epoch]
        y_val_true = self.val_labels[epoch]

        # Convert tensors to CPU numpy arrays if needed
        def to_numpy(x):
            if hasattr(x, 'detach'):
                return x.detach().cpu().numpy()
            return x

        y_train_pred = to_numpy(y_train_pred)
        y_train_true = to_numpy(y_train_true)
        y_val_pred = to_numpy(y_val_pred)
        y_val_true = to_numpy(y_val_true)

        # Compute ROC curve and ROC area for train
        fpr_train, tpr_train, _ = roc_curve(y_train_true, y_train_pred)
        roc_auc_train = auc(fpr_train, tpr_train)

        # Compute ROC curve and ROC area for validation
        fpr_val, tpr_val, _ = roc_curve(y_val_true, y_val_pred)
        roc_auc_val = auc(fpr_val, tpr_val)

        # Plotting
        fontsize = 16
        plt.figure(figsize=(7, 5))
        plt.plot(fpr_train, tpr_train, linestyle='--', label=f'Train ROC (AUC = {roc_auc_train:.3f})')
        plt.plot(fpr_val, tpr_val, linestyle='-', label=f'Val   ROC (AUC = {roc_auc_val:.3f})')
        # plt.plot([0, 1], [0, 1], color='grey', linestyle=':', label='Chance')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=fontsize)
        plt.ylabel('True Positive Rate', fontsize=fontsize)
        plt.title(f'ROC Curves at Epoch {epoch}', fontsize=fontsize+2)
        plt.legend(loc='lower right', fontsize=fontsize)
        plt.tight_layout()

        # Show or save
        if show:
            plt.show()
            # Ensure the directory exists
        output_path = os.path.join(path, "roc_auc", file_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)

        # return {'fpr_train': fpr_train, 'tpr_train': tpr_train, 'auc_train': roc_auc_train,
        #         'fpr_val': fpr_val, 'tpr_val': tpr_val, 'auc_val': roc_auc_val}



    def plot_accuracy(self, file_name="acc.png", show=True):
        """
        Plot training and validation accuracy for binary classification.
        """

        train_acc = [a.cpu().item() if hasattr(a, "cpu") else a for a in self.train_acc]
        val_acc = [a.cpu().item() if hasattr(a, "cpu") else a for a in self.val_acc]
        tr_err = [e.cpu().item() if hasattr(e, "cpu") else e for e in self.train_acc_err]
        vl_err = [e.cpu().item() if hasattr(e, "cpu") else e for e in self.val_acc_err]

        if hasattr(self, "epoch_warmstart") and self.epoch_warmstart and self.epochs[0] != 0:
            x = list(range(self.epoch_warmstart)) + self.epochs
        else:
            x = self.epochs

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(x, train_acc, yerr=tr_err, label="Train Accuracy", marker='o')
        ax.errorbar(x, val_acc, yerr=vl_err, label="Validation Accuracy", marker='o')

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Training and Validation Accuracy")
        ax.grid(True)
        ax.legend()

        fig.tight_layout()
        if show:
            plt.show()
        fig.savefig(file_name)


    def plot_efficiency(self, file_name="eff.png", show=True):
        """
        Plot training and validation efficiency for binary classification.
        """

        train_eff = [e.cpu().item() if hasattr(e, "cpu") else e for e in self.train_eff]
        val_eff = [e.cpu().item() if hasattr(e, "cpu") else e for e in self.val_eff]
        tr_err = [e.cpu().item() if hasattr(e, "cpu") else e for e in self.train_eff_err]
        vl_err = [e.cpu().item() if hasattr(e, "cpu") else e for e in self.val_eff_err]

        if hasattr(self, "epoch_warmstart") and self.epoch_warmstart and self.epochs[0] != 0:
            x = list(range(self.epoch_warmstart)) + self.epochs
        else:
            x = self.epochs

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(x, train_eff, yerr=tr_err, label="Train Efficiency", marker='o')
        ax.errorbar(x, val_eff, yerr=vl_err, label="Validation Efficiency", marker='o')

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Efficiency")
        ax.set_title("Training and Validation Efficiency")
        ax.grid(True)
        ax.legend()

        fig.tight_layout()
        if show:
            plt.show()
        fig.savefig(file_name)


    def plot_rejection(self, file_name="rej.png", show=True):
        """
        Plot training and validation rejection for binary classification.
        """

        train_rej = [r.cpu().item() if hasattr(r, "cpu") else r for r in self.train_rej]
        val_rej = [r.cpu().item() if hasattr(r, "cpu") else r for r in self.val_rej]
        tr_err = [e.cpu().item() if hasattr(e, "cpu") else e for e in self.train_rej_err]
        vl_err = [e.cpu().item() if hasattr(e, "cpu") else e for e in self.val_rej_err]

        if hasattr(self, "epoch_warmstart") and self.epoch_warmstart and self.epochs[0] != 0:
            x = list(range(self.epoch_warmstart)) + self.epochs
        else:
            x = self.epochs

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(x, train_rej, yerr=tr_err, label="Train Rejection", marker='o')
        ax.errorbar(x, val_rej, yerr=vl_err, label="Validation Rejection", marker='o')

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Rejection")
        ax.set_title("Training and Validation Rejection")
        ax.grid(True)
        ax.legend()

        fig.tight_layout()
        if show:
            plt.show()
        fig.savefig(file_name)
