from abc import ABC, abstractmethod
import torch
import matplotlib.pyplot as plt
import os


class Trainer(ABC):
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
            train_acc, val_acc (List[Tensor]): Per‐epoch class accuracies.
            train_loss, val_loss (List[float]): Per‐epoch loss values.
            epochs (List[int]): Epoch indices logged.
        """
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
        self.LCA_classes = config.get('model.LCA_classes')

    @abstractmethod
    def eval_one_epoch(self, train=True):
        """
        Run a single epoch of evaluation (and training if train=True).

        Must:
          - Iterate over the appropriate loader.
          - Compute batch‐wise loss and per‐class accuracy.
          - Return a tuple (avg_loss: float, class_acc: Tensor).

        Args:
            train (bool):
                If True, perform backprop through train_loader;
                if False, evaluate on val_loader.

        Returns:
            avg_loss (float): Average loss over all batches.
            class_acc (Tensor): Per‐class accuracy for this epoch.
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
        Save the model's state dict to disk.

        Args:
            file_name (str): Path to save the `.pth` or `.pt` file.
        """
        torch.save(self.model.state_dict(), file_name)
        if save_config: # print config file
            print("Saving config file as txt file")
            self.config.print(file_name.replace('.pt','.txt'))

    def save_dataframe(self, file_name):
        """
        Export training history to a CSV or DataFrame file.

        Subclasses may override to customize the exact format.

        Args:
            file_name (str): Path to save the metrics (e.g. `.csv`).
        """
        pass

    def plot_loss(self, file_name="loss.png", show=True):
        """
        Plot and optionally display training vs. validation loss.

        Args:
            file_name (str): File path for saving the figure.
            show (bool): If True, display the plot with plt.show().
        """
        import matplotlib.pyplot as plt
        import os

        plt.plot(self.train_loss, label="Train Loss")
        plt.plot(self.val_loss, label="Validation Loss")

        plt.xlabel('epoch')
        plt.ylabel('Cross Entropy Loss')

        plt.legend()
        if show:
            plt.show()
        plt.savefig(file_name)

    def plot_accuracy(self, file_name="acc.png", show=True):
        """
        Plot per‐class accuracy for training and validation.

        Assumes exactly 4 classes (indices 0–3). Adjust slicing logic
        if you have a different number of classes.

        Args:
            file_name (str): File path for saving the figure.
            show (bool): If True, display the plots with plt.show().
        """

        class_acc_vl = {f"class{i}_acc_vl" : [] for i in range(self.LCA_classes)}
        class_acc_vl_err = {f"class{i}_acc_vl_err" : [] for i in range(self.LCA_classes)}
        
        for i in range(self.LCA_classes):
            for vl_acc,vl_acc_err in zip(self.val_acc,self.val_acc_err):
                class_acc_vl[f"class{i}_acc_vl"].append(vl_acc[i])
                class_acc_vl_err[f"class{i}_acc_vl_err"].append(vl_acc_err[i])

        class_acc_tr = {f"class{i}_acc_tr" : [] for i in range(self.LCA_classes)}
        class_acc_tr_err = {f"class{i}_acc_tr_err" : [] for i in range(self.LCA_classes)}
        
        for i in range(self.LCA_classes):
            for tr_acc,tr_acc_err in zip(self.train_acc,self.train_acc_err):
                class_acc_tr[f"class{i}_acc_tr"].append(tr_acc[i])
                class_acc_tr_err[f"class{i}_acc_tr_err"].append(tr_acc_err[i])

        fig, axarr = plt.subplots(1, 2, figsize=(10, 5))

        for i in range(self.LCA_classes):
            axarr[0].errorbar(self.epochs, class_acc_tr[f"class{i}_acc_tr"], yerr=class_acc_tr_err[f"class{i}_acc_tr_err"], label=f"LCA={i}")
            axarr[1].errorbar(self.epochs, class_acc_vl[f"class{i}_acc_vl"], yerr=class_acc_vl_err[f"class{i}_acc_vl_err"], label=f"LCA={i}")

        axarr[0].set_xlabel('epoch')
        axarr[0].set_ylabel('training accuracy')
        
        axarr[0].legend()

        axarr[1].set_xlabel('epoch')
        axarr[1].set_ylabel('validation accuracy')
        axarr[1].legend()

        fig.tight_layout()
        if show:
            plt.show()
        plt.savefig(file_name)

    def plot_efficiency(self, file_name="eff.png", show=True):

        class_eff_vl = {f"class{i}_eff_vl" : [] for i in range(self.LCA_classes)}
        
        for i in range(self.LCA_classes):
            for vl_eff in self.val_eff:
                class_eff_vl[f"class{i}_eff_vl"].append(vl_eff[i])

        class_eff_tr = {f"class{i}_eff_tr" : [] for i in range(self.LCA_classes)}

        for i in range(self.LCA_classes):
            for tr_eff in self.train_eff:
                class_eff_tr[f"class{i}_eff_tr"].append(tr_eff[i])

        fig, axarr = plt.subplots(1, 2, figsize=(10, 5))

        for i in range(self.LCA_classes):
            axarr[0].plot(class_eff_tr[f"class{i}_eff_tr"], label=f"LCA={i}")
            axarr[1].plot(class_eff_vl[f"class{i}_eff_vl"], label=f"LCA={i}")

        axarr[0].set_xlabel('epoch')
        axarr[0].set_ylabel('training efficiency')
        axarr[0].legend()

        axarr[1].set_xlabel('epoch')
        axarr[1].set_ylabel('validation efficiency')
        axarr[1].legend()

        fig.tight_layout()
        if show:
            plt.show()
        plt.savefig(file_name)
    
    def plot_rejection(self, file_name="rej.png", show=True):

        class_rej_vl = {f"class{i}_rej_vl" : [] for i in range(self.LCA_classes)}
        
        for i in range(self.LCA_classes):
            for vl_rej in self.val_rej:
                class_rej_vl[f"class{i}_rej_vl"].append(vl_rej[i])

        class_rej_tr = {f"class{i}_rej_tr" : [] for i in range(self.LCA_classes)}

        for i in range(self.LCA_classes):
            for tr_rej in self.train_rej:
                class_rej_tr[f"class{i}_rej_tr"].append(tr_rej[i])

        fig, axarr = plt.subplots(1, 2, figsize=(10, 5))

        for i in range(self.LCA_classes):
            axarr[0].plot(class_rej_tr[f"class{i}_rej_tr"], label=f"LCA={i}")
            axarr[1].plot(class_rej_vl[f"class{i}_rej_vl"], label=f"LCA={i}")

        axarr[0].set_xlabel('epoch')
        axarr[0].set_ylabel('training rejection')
        axarr[0].legend()

        axarr[1].set_xlabel('epoch')
        axarr[1].set_ylabel('validation rejection')
        axarr[1].legend()

        fig.tight_layout()
        if show:
            plt.show()
        plt.savefig(file_name)