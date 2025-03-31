from abc import ABC, abstractmethod
import torch
import matplotlib.pyplot as plt

class Trainer(ABC):

    def __init__(self, config, model, train_loader, val_loader):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_acc = []
        self.val_acc = []
        self.train_loss = []
        self.val_loss = []
        self.epochs = []
        self.LCA_classes = config.get('model.LCA_classes')

    @abstractmethod
    def eval_one_epoch(self, train=True):
        pass

    @abstractmethod
    def train(self, epochs=10, learning_rate=0.001):
        pass

    def save_model(self, file_name):
        torch.save(self.model.state_dict(), file_name)

    def save_dataframe(self, file_name):
        pass

    def plot_loss(self, file_name="loss.png", show=True):
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

        class_acc_vl = {f"class{i}_acc_vl" : [] for i in range(self.LCA_classes)}
        #class1_acc_vl = []
        #class2_acc_vl = []
        #class3_acc_vl = []
        #class4_acc_vl = []
        
        for i in range(self.LCA_classes):
            for vl_acc in self.val_acc:
                class_aa[f"class{i}_acc_vl"].append(vl_acc[i])

        #for vl_acc in self.val_acc:
        #    class1_acc_vl.append(vl_acc[0])
        #    class2_acc_vl.append(vl_acc[1])
        #    class3_acc_vl.append(vl_acc[2])
        #    class4_acc_vl.append(vl_acc[3])

        class_acc_tr = {f"class{i}_acc_tr" : [] for i in range(self.LCA_classes)}
        #class1_acc_tr = []
        #class2_acc_tr = []
        #class3_acc_tr = []
        #class4_acc_tr = []
        
        for i in range(self.LCA_classes):
            for tr_acc in self.train_acc:
                class_acc_tr[f"class{i}_acc_tr"].append(tr_acc[i])
        #for tr_acc in self.train_acc:
        #    class1_acc_tr.append(tr_acc[0])
        #    class2_acc_tr.append(tr_acc[1])
        #    class3_acc_tr.append(tr_acc[2])
        #    class4_acc_tr.append(tr_acc[3])

        fig, axarr = plt.subplots(1, 2, figsize=(10, 5))

        for i in range(self.LCA_classes):
            axarr[0].plot(class_acc_tr[f"class{i}_acc_tr"], label=f"LCA={i}")
            axarr[1].plot(class_acc_vl[f"class{i}_acc_vl"], label=f"LCA={i}")
            
        #axarr[0].plot(class1_acc_tr, label="LCA=0")
        #axarr[0].plot(class2_acc_tr, label="LCA=1")
        #axarr[0].plot(class3_acc_tr, label="LCA=2")
        #axarr[0].plot(class4_acc_tr, label="LCA=3")

        axarr[0].set_xlabel('epoch')
        axarr[0].set_ylabel('training accuracy')

        axarr[0].legend()

        #axarr[1].plot(class1_acc_vl, label="LCA=0")
        #axarr[1].plot(class2_acc_vl, label="LCA=1")
        #axarr[1].plot(class3_acc_vl, label="LCA=2")
        #axarr[1].plot(class4_acc_vl, label="LCA=3")

        axarr[1].set_xlabel('epoch')
        axarr[1].set_ylabel('validation accuracy')

        axarr[1].legend()

        fig.tight_layout()
        if show:
            plt.show()
        plt.savefig(file_name)

