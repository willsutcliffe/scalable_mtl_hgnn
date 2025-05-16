import pytorch_lightning as L
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from collections import defaultdict

import torch
from torch import nn
from torch_scatter import scatter_add

from wmpgnn.util.functions import acc_four_class


class HGNNLightningModule(L.LightningModule):
    def __init__(self, model, optimizer_class, optimizer_params, pos_weights):
        super().__init__()

        self.model = model

        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params

        # Loss functions
        self.LCA_criterion = nn.CrossEntropyLoss(weight=pos_weights["LCA"])
        self.t_nodes_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights["t_nodes"])
        self.tt_edges_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights["tt_edges"])
        self.tPV_edges_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([6.1118]))  # Average no. of PVs for each track one PV is correct

        # trn, val logging
        self.trn_log = {"LCA_loss": [], "t_nodes_loss": [], "tt_edges_loss": [], "tPV_edges_loss": [], "combined_loss": [],
                        "LCA0_acc": [], "LCA1_acc": [], "LCA2_acc": [], "LCA3_acc": [],
                        "tPV_edge_acc": [], "PV_has_B_acc": []}
        self.val_log = {"LCA_loss": [], "t_nodes_loss": [], "tt_edges_loss": [], "tPV_edges_loss": [], "combined_loss": [],
                        "LCA0_acc": [], "LCA1_acc": [], "LCA2_acc": [], "LCA3_acc": [],
                        "tPV_edge_acc": [], "PV_has_B_acc": []}

    def forward(self, batch):
        return self.model(batch)
    
    def configure_optimizers(self):
        return self.optimizer_class(self.model.parameters(), **self.optimizer_params)
    
    # def on_train_epoch_start(self):
    #     self.trainer.train_dataloader.sampler.set_epoch(self.current_epoch)

    def shared_step(self, batch, log_dict):  # Runs 1 batch
        loss_t_nodes = 0.
        loss_tt_edges = 0.
        loss_tPV_edges = 0.

        """Truth information for interference"""
        y_tt_LCA = batch[('tracks', 'to', 'tracks')].y.argmax(dim=1)  # for LCA multi classification
        y_tPV_edges = batch[('tracks', 'to', 'PVs')].y.to(dtype=torch.float32)  # PV to track edge classification
        y_tt_edges = (batch[('tracks', 'to', 'tracks')].y[:, 0] == 0).unsqueeze(1).float()  # for edge pruning, wether edge exists or not
        
        # Getting the truth for node pruning
        num_nodes = batch['tracks'].x.shape[0]
        out = batch[('tracks', 'to', 'tracks')].edges.new_zeros(num_nodes, batch[('tracks', 'to', 'tracks')].y.shape[1])
        node_sum = scatter_add(batch[('tracks', 'to', 'tracks')].y, batch[('tracks', 'to', 'tracks')].edge_index[0], out=out, dim=0)
        y_t_nodes = ((torch.sum(node_sum[:, 1:], 1) > 0)).unsqueeze(1).float()  # node pruning label

        # General track to PV, TODO need to check it, I think its about finding the PV which produce B
        yb = y_t_nodes[batch[('tracks', 'to', 'pvs')]['edge_index'][0]] * batch[('tracks', 'to', 'pvs')].y
        pv_sum = scatter_add(yb, batch[('tracks', 'to', 'pvs')].edge_index[1], dim=0)

        """Passing to the model"""
        outputs = self.model(batch)

        """Getting the loss"""
        loss_LCA = self.LCA_criterion(outputs[('tracks', 'to', 'tracks')].edges, y_tt_LCA)
        # Looping over all the GN blocks to grab the interference
        for block in self.model._blocks:
            loss_t_nodes += self.t_nodes_criterion(block.node_logits['tracks'], y_t_nodes)
            loss_tt_edges += self.tt_edges_criterion(block.edge_logits[('tracks', 'to', 'tracks')], y_tt_edges)
            loss_tPV_edges += self.tPV_edges_criterion(block.edge_logits[('tracks', 'to', 'PVs')], y_tPV_edges)

        """Combing the loss"""
        loss = (  # Here we can add the weights
                loss_LCA +
                loss_t_nodes +
                loss_tt_edges +
                loss_tPV_edges
        )

        """Getting the accuracy"""
        acc_LCA = acc_four_class(outputs[('tracks', 'to', 'tracks')].edges, y_tt_LCA)
        acc_tPV_edge = torch.sum(y_tPV_edges == (self.model._blocks[-1].edge_weights[('tracks', 'to', 'pvs')] > 0.5)) / y_tPV_edges.shape[0]
        pv_target = 1. * (pv_sum > 0)
        acc_PV_has_B = torch.sum(pv_target == (self.model._blocks[-1].node_weights['pvs'] > 0.5)) / pv_target.shape[0]

        """Logging"""
        log_dict["LCA_loss"].append(loss_LCA.item())
        log_dict["t_nodes_loss"].append(loss_t_nodes.item())
        log_dict["tt_edges_loss"].append(loss_tt_edges.item())
        log_dict["tPV_edges_loss"].append(loss_tPV_edges.item())
        log_dict["combined_loss"].append(loss.item())

        log_dict["LCA0_acc"].append(acc_LCA[0])
        log_dict["LCA1_acc"].append(acc_LCA[1])
        log_dict["LCA2_acc"].append(acc_LCA[2])
        log_dict["LCA3_acc"].append(acc_LCA[3])
        log_dict["tPV_edge_acc"].append(acc_tPV_edge)
        log_dict["PV_has_B_acc"].append(acc_PV_has_B)

        return loss
        

    def training_step(self, batch, batch_idx): 
        loss = self.shared_step(batch, log_dict=self.trn_log)
        return loss
    
    def validation_step(self, batch, batch_idx): 
        loss = self.shared_step(batch, log_dict=self.val_log)
        import pdb; pdb.set_trace()
        return loss
    
    def on_train_epoch_end(self):
        avg_losses = {key: torch.tensor(vals).nanmean(dim=0) for key, vals in self.trn_log.items()}
        for key, val in avg_losses.items():
            self.log(f"train/{key}", val, prog_bar=True, on_epoch=True, on_step=False)
        self.trn_log = defaultdict(list)

    def on_validation_epoch_end(self):
        avg_losses = {key: torch.tensor(vals).nanmean(dim=0) for key, vals in self.val_log.items()}
        for key, val in avg_losses.items():
            self.log(f"val/{key}", val, prog_bar=True, on_epoch=True, on_step=False)
        self.val_log = defaultdict(list)


# Here is a trainer wrapper
def training(model, pos_weight, epochs, n_gpu, trn_loader, val_loader):
    module = HGNNLightningModule(
        model=model,
        pos_weights=pos_weight,
        optimizer_class=torch.optim.Adam,
        optimizer_params={"lr": 1e-3}
    )

    early_stopping = EarlyStopping(
        monitor="val/combined_loss",
        mode="min",
        patience=5,
    )

    best_model_callback = ModelCheckpoint(
        filename="best-{epoch:02d}-{val_combined_loss:.2f}",
        monitor="val/combined_loss",
        mode="min",
        save_top_k=1
    )

    all_epochs_callback = ModelCheckpoint(
        filename="epoch-{epoch:02d}",
        save_top_k=-1,
        every_n_epochs=1
    )

    trainer = Trainer(
        max_epochs=epochs,
        accelerator='gpu',
        devices=n_gpu,
        strategy="auto",  # ddp_notebook change it to normal ddp
        callbacks=[early_stopping, best_model_callback, all_epochs_callback],
        precision="32",  # never doe 16-mixed
        gradient_clip_val=1.0
    )

    """Start training"""
    trainer.fit(module, trn_loader, val_loader)