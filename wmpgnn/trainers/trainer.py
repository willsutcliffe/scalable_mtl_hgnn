from abc import ABC, abstractmethod
import torch
import matplotlib.pyplot as plt
import os

class Trainer(ABC):

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
        self.LCA_classes = config.get('model.LCA_classes')
        self.epoch_warmstart = 0
        

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
        history['bce_nodes_train_loss'] = self.bce_nodes_train_loss
        history['bce_nodes_val_loss']   = self.bce_nodes_val_loss
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
        self.bce_nodes_train_loss = history['bce_nodes_train_loss']
        self.bce_nodes_val_loss   = history['bce_nodes_val_loss']
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

    def plot_accuracy(self, file_name="acc.png", show=True):

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

        if self.epochs[0] != 0:
            x = [e for e in range(self.epoch_warmstart)]+self.epochs
        else:
            x = self.epochs
        for i in range(self.LCA_classes):
            axarr[0].errorbar(x, class_acc_tr[f"class{i}_acc_tr"], yerr=class_acc_tr_err[f"class{i}_acc_tr_err"], label=f"LCA={i}")
            axarr[1].errorbar(x, class_acc_vl[f"class{i}_acc_vl"], yerr=class_acc_vl_err[f"class{i}_acc_vl_err"], label=f"LCA={i}")

        axarr[0].set_xlabel('epoch')
        axarr[0].set_ylabel('training accuracy')
        axarr[0].grid()
        axarr[0].legend()

        axarr[1].set_xlabel('epoch')
        axarr[1].set_ylabel('validation accuracy')
        axarr[1].grid()
        axarr[1].legend()

        fig.tight_layout()
        if show:
            plt.show()
        plt.savefig(file_name)

    def plot_efficiency(self, file_name="eff.png", show=True):

        class_eff_vl = {f"class{i}_eff_vl" : [] for i in range(self.LCA_classes)}
        class_eff_vl_err = {f"class{i}_eff_vl_err" : [] for i in range(self.LCA_classes)}
        
        for i in range(self.LCA_classes):
            for vl_eff,vl_eff_err in zip(self.val_eff,self.val_eff_err):
                class_eff_vl[f"class{i}_eff_vl"].append(vl_eff[i])
                class_eff_vl_err[f"class{i}_eff_vl_err"].append(vl_eff_err[i])

        class_eff_tr = {f"class{i}_eff_tr" : [] for i in range(self.LCA_classes)}
        class_eff_tr_err = {f"class{i}_eff_tr_err" : [] for i in range(self.LCA_classes)}
        
        for i in range(self.LCA_classes):
            for tr_eff,tr_eff_err in zip(self.train_eff,self.train_eff_err):
                class_eff_tr[f"class{i}_eff_tr"].append(tr_eff[i])
                class_eff_tr_err[f"class{i}_eff_tr_err"].append(tr_eff_err[i])

        fig, axarr = plt.subplots(1, 2, figsize=(10, 5))

        if self.epochs[0] != 0:
            x = [e for e in range(self.epoch_warmstart)]+self.epochs
        else:
            x = self.epochs
        for i in range(self.LCA_classes):
            axarr[0].errorbar(x, class_eff_tr[f"class{i}_eff_tr"], yerr=class_eff_tr_err[f"class{i}_eff_tr_err"], label=f"LCA={i}")
            axarr[1].errorbar(x, class_eff_vl[f"class{i}_eff_vl"], yerr=class_eff_vl_err[f"class{i}_eff_vl_err"], label=f"LCA={i}")
            
        axarr[0].set_xlabel('epoch')
        axarr[0].set_ylabel('training efficiency')
        axarr[0].grid()
        axarr[0].legend()

        axarr[1].set_xlabel('epoch')
        axarr[1].set_ylabel('validation efficiency')
        axarr[1].grid()
        axarr[1].legend()

        fig.tight_layout()
        if show:
            plt.show()
        plt.savefig(file_name)
    
    def plot_rejection(self, file_name="rej.png", show=True):

        class_rej_vl = {f"class{i}_rej_vl" : [] for i in range(self.LCA_classes)}
        class_rej_vl_err = {f"class{i}_rej_vl_err" : [] for i in range(self.LCA_classes)}
        
        for i in range(self.LCA_classes):
            for vl_rej,vl_rej_err in zip(self.val_rej,self.val_rej_err):
                class_rej_vl[f"class{i}_rej_vl"].append(vl_rej[i])
                class_rej_vl_err[f"class{i}_rej_vl_err"].append(vl_rej_err[i])

        class_rej_tr = {f"class{i}_rej_tr" : [] for i in range(self.LCA_classes)}
        class_rej_tr_err = {f"class{i}_rej_tr_err" : [] for i in range(self.LCA_classes)}
        
        for i in range(self.LCA_classes):
            for tr_rej,tr_rej_err in zip(self.train_rej,self.train_rej_err):
                class_rej_tr[f"class{i}_rej_tr"].append(tr_rej[i])
                class_rej_tr_err[f"class{i}_rej_tr_err"].append(tr_rej_err[i])

        fig, axarr = plt.subplots(1, 2, figsize=(10, 5))

        if self.epochs[0] != 0:
            x = [e for e in range(self.epoch_warmstart)]+self.epochs
        else:
            x = self.epochs
        for i in range(self.LCA_classes):
            axarr[0].errorbar(x, class_rej_tr[f"class{i}_rej_tr"], yerr=class_rej_tr_err[f"class{i}_rej_tr_err"], label=f"LCA={i}")
            axarr[1].errorbar(x, class_rej_vl[f"class{i}_rej_vl"], yerr=class_rej_vl_err[f"class{i}_rej_vl_err"], label=f"LCA={i}")

        axarr[0].set_xlabel('epoch')
        axarr[0].set_ylabel('training rejection')
        axarr[0].grid()
        axarr[0].legend()

        axarr[1].set_xlabel('epoch')
        axarr[1].set_ylabel('validation rejection')
        axarr[1].grid()
        axarr[1].legend()

        fig.tight_layout()
        if show:
            plt.show()
        plt.savefig(file_name)