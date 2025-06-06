import pytorch_lightning as L
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from collections import defaultdict
import copy

import pandas as pd
import numpy as np

import torch
torch.set_float32_matmul_precision("high")
from torch import nn
from torch_scatter import scatter_add

from helper import make_loggable, eval_reco_performance, get_ref_signal
from plot_helper import plot_gn_block_dist, plot_ft_nodes
from wmpgnn.util.functions import acc_four_class


class HGNNLightningModule(L.LightningModule):
    def __init__(self, model, optimizer_class, optimizer_params, pos_weights, version=0, ref_signal=''):
        super().__init__()
        self.save_hyperparameters({
            "pos_weights": make_loggable(pos_weights),
        })

        self.model = model

        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params

        # Loss functions
        self.LCA_criterion = nn.CrossEntropyLoss(weight=pos_weights["LCA"])
        self.t_nodes_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights["t_nodes"])
        self.frag_nodes_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights["frag"])
        self.ft_nodes_criterion = nn.CrossEntropyLoss(weight=pos_weights["FT"])
        self.tt_edges_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights["tt_edges"])
        self.tPV_edges_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([6.1118]))  # Average no. of PVs for each track one PV is correct

        # trn, val, tst logging
        self.trn_log = {"LCA_loss": [], "t_nodes_loss": [], "frag_loss": [], "ft_loss": [], "tt_edges_loss": [], "tPV_edges_loss": [], "combined_loss": [],
                        "tPV_edge_acc": [], "PV_has_B_acc": []}
        self.val_log = {"LCA_loss": [], "t_nodes_loss": [], "frag_loss": [], "ft_loss": [], "tt_edges_loss": [], "tPV_edges_loss": [], "combined_loss": [],
                        "tPV_edge_acc": [], "PV_has_B_acc": []}
        self.tst_log = {"ft_y": torch.tensor([]),
                        "tPV_edge_acc": [], "PV_has_B_acc": []}
        for i in range(4):
            self.trn_log[f"LCA_class{i}_num"] = []
            self.val_log[f"LCA_class{i}_num"] = []
            self.tst_log[f"LCA_class{i}_num"] = []
            for j in range(4):
                self.trn_log[f"LCA_class{i}_pred_class{j}"] = []
                self.val_log[f"LCA_class{i}_pred_class{j}"] = []
                self.tst_log[f"LCA_class{i}_pred_class{j}"] = []
        for i in range(6):
            self.trn_log[f"w_{i}"] = []
            self.val_log[f"w_{i}"] = []

        # Eval flags on tst sample which can be turned on and off
        self.version = version
        self.ref_signal = get_ref_signal(ref_signal)
        self.get_node_performance = True  # HOnestly this hsould be always true like everything but what ever...
        self.get_edge_performance = True
        self.get_PV_performance = True
        self.get_reco_performance = True
        self.get_frag_performance = True

    
    def init_tst_log(self):
        # Add tensors to dict for logging
        for i in range(len(self.model._blocks)):
            if self.get_node_performance:
                self.tst_log[f"sig_nodes_score_{i}"] = torch.tensor([])
                self.tst_log[f"bkg_nodes_score_{i}"] = torch.tensor([])
            if self.get_edge_performance:
                self.tst_log[f"sig_edges_score_{i}"] = torch.tensor([])
                self.tst_log[f"bkg_edges_score_{i}"] = torch.tensor([])
            if self.get_frag_performance:
                self.tst_log[f"frag_pos_part_score_{i}"] = torch.tensor([])
                self.tst_log[f"frag_neg_part_score_{i}"] = torch.tensor([])
            if self.get_reco_performance:
                self.tst_log[f"ft_score_{i}"] = torch.tensor([])
                self.tst_log[f"bbar_ft_score_{i}"] = torch.tensor([])
                self.tst_log[f"none_ft_score_{i}"] = torch.tensor([])
                self.tst_log[f"b_ft_score_{i}"] = torch.tensor([])

        # Initiate the dataframes for reco performance eval
        self.signal_df = pd.DataFrame(
        columns=['EventNumber', 'NumParticlesInEvent', 'NumSignalParticles', 'NumBkgParticles_noniso', 
                'PerfectSignalReconstruction', 'AllParticles', 'PerfectReco', 'NoneIso', 'PartReco', 'NotFound', 'SigMatch', 
                'B_id', 'Pred_FT'])
        self.signal_df = self.signal_df.astype(
            {'EventNumber': np.int32, 'NumParticlesInEvent': np.int32, 'NumSignalParticles': np.int32, 'NumBkgParticles_noniso': np.int32,
                'PerfectSignalReconstruction': np.int32,
                'AllParticles': np.int32, 'PerfectReco': np.int32,
                'NoneIso': np.int32, 'PartReco': np.int32, 'NotFound': np.int32, 'SigMatch': np.int32, 'B_id': np.int32, 'Pred_FT': np.int32})

        self.event_df = pd.DataFrame(
            columns=['EventNumber', 'NumParticlesInEvent', 'NumParticlesFromHeavyHadronInEvent',
                    'NumBackgroundParticlesInEvent', 'NumSelectedParticlesInEvent',
                    'NumSelectedParticlesFromHeavyHadronInEvent',
                    'NumSelectedBackgroundParticlesInEvent', 'NumTruthClustersGen1', 'NumTruthClustersGen2',
                    'NumTruthClustersGen3', 'NumTruthClustersGen4', 'NumRecoClustersGen1', 'NumRecoClustersGen2',
                    'NumRecoClustersGen3', 'NumRecoClustersGen4', 'MaxTruthFullChainDepthInEvent',
                    'EfficiencyParticlesFromHeavyHadronInEvent', 'EfficiencyBackgroundParticlesInEvent',
                    'BackgroundRejectionPowerInEvent', 'PerfectEventReconstruction', 'TimeNodeFiltering',
                    'TimeEdgeFiltering',
                    'TimeLCAReconstruction', 'TimeSequence', 'NumTrueSignalsInEvent', 'NumRecoSignalsInEvent',
                    'TimeModel', 'TimeReco', 'TimeTruth'])


    def forward(self, batch):
        return self.model(batch)
    

    def configure_optimizers(self):
        return self.optimizer_class(self.model.parameters(), **self.optimizer_params)
    
    # def on_train_epoch_start(self):
    #     self.trainer.train_dataloader.sampler.set_epoch(self.current_epoch)

    def shared_step(self, batch, batch_idx, log_dict):  # Runs 1 batch
        loss_t_nodes = 0.
        loss_tt_edges = 0.
        loss_tPV_edges = 0.
        loss_frag_nodes = 0.
        loss_ft_nodes = 0.

        """Truth information for interference"""
        y_tt_LCA = batch[('tracks', 'to', 'tracks')].y.argmax(dim=1)  # for LCA multi classification
        y_tPV_edges = batch[('tracks', 'to', 'pvs')].y.to(dtype=torch.float32)  # PV to track edge classification
        y_tt_edges = (batch[('tracks', 'to', 'tracks')].y[:, 0] == 0).unsqueeze(1).float()  # for edge pruning, wether edge exists or not
        y_frag = (batch['tracks'].frag !=0).float().unsqueeze(1)
        y_ft = batch['tracks'].ft + 1  # shifted by one 0 = bbar, 1 = none, 2 = b

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
        # print(torch.cuda.memory_allocated() / (1024 ** 2) )
        # print(torch.cuda.memory_summary())

        """Getting the loss"""
        loss_LCA = self.LCA_criterion(outputs[('tracks', 'to', 'tracks')].edges, y_tt_LCA)
        # Looping over all the GN blocks to grab the interference
        for block in self.model._blocks:
            loss_t_nodes += self.t_nodes_criterion(block.node_logits['tracks'], y_t_nodes)
            loss_frag_nodes += self.frag_nodes_criterion(block.node_logits['frag'], y_frag)
            loss_ft_nodes += self.ft_nodes_criterion(block.node_logits['ft'], y_ft)
            loss_tt_edges += self.tt_edges_criterion(block.edge_logits[('tracks', 'to', 'tracks')], y_tt_edges)
            loss_tPV_edges += self.tPV_edges_criterion(block.edge_logits[('tracks', 'to', 'pvs')], y_tPV_edges)
        
        """Combing the loss"""
        w = torch.exp(-self.model.loss_weights)
        loss = 1/2 * (
                w[0] * loss_LCA +
                w[1] * loss_t_nodes +
                w[2] * loss_tt_edges +
                w[3] * loss_tPV_edges + 
                w[4] * loss_frag_nodes +
                w[5] * loss_ft_nodes +
                torch.sum(self.model.loss_weights)
        )

        """Getting the accuracy"""
        acc_LCA = acc_four_class(outputs[('tracks', 'to', 'tracks')].edges, y_tt_LCA)
        acc_tPV_edge = torch.sum(y_tPV_edges == (self.model._blocks[-1].edge_weights[('tracks', 'to', 'pvs')] > 0.5)) / y_tPV_edges.shape[0]
        pv_target = 1. * (pv_sum > 0)
        acc_PV_has_B = torch.sum(pv_target == (self.model._blocks[-1].node_weights['pvs'] > 0.5)) / pv_target.shape[0]

        """Logging"""
        log_dict["LCA_loss"].append(loss_LCA.item())
        log_dict["t_nodes_loss"].append(loss_t_nodes.item())
        log_dict["frag_loss"].append(loss_frag_nodes.item())
        log_dict["ft_loss"].append(loss_ft_nodes.item())
        log_dict["tt_edges_loss"].append(loss_tt_edges.item()) 
        log_dict["tPV_edges_loss"].append(loss_tPV_edges.item())
        log_dict["combined_loss"].append(loss.item())
        for key, values in acc_LCA.items():
            log_dict[key].append(values)
        log_dict["tPV_edge_acc"].append(acc_tPV_edge)
        log_dict["PV_has_B_acc"].append(acc_PV_has_B)
        for i, weight in enumerate(self.model.loss_weights):
            log_dict[f"w_{i}"].append(weight.item())

        return loss
        

    def training_step(self, batch, batch_idx): 
        loss = self.shared_step(batch, batch_idx, log_dict=self.trn_log)
        return loss

    
    def validation_step(self, batch, batch_idx): 
        loss = self.shared_step(batch, batch_idx,log_dict=self.val_log)
        return loss


    def test_step(self, batch, batch_idx):
        """Getting the true value"""
        # Variables for accuracy calculation
        y_tt_LCA = batch[('tracks', 'to', 'tracks')].y.argmax(dim=1)  # for LCA multi classification
        y_tPV_edges = batch[('tracks', 'to', 'pvs')].y.to(dtype=torch.float32)
        y_frag = (batch['tracks'].frag !=0).float().unsqueeze(1)
        y_ft = batch['tracks'].ft + 1
        self.tst_log["ft_y"] = torch.cat([self.tst_log["ft_y"], y_ft.cpu()], dim=0)

        num_nodes = batch['tracks'].x.shape[0]
        out = batch[('tracks', 'to', 'tracks')].edges.new_zeros(num_nodes, batch[('tracks', 'to', 'tracks')].y.shape[1])
        node_sum = scatter_add(batch[('tracks', 'to', 'tracks')].y, batch[('tracks', 'to', 'tracks')].edge_index[0], out=out, dim=0)
        y_t_nodes = ((torch.sum(node_sum[:, 1:], 1) > 0))  # node pruning labels
        y_tt_edges = (batch[('tracks', 'to', 'tracks')].y[:, 0] == 0)  # edge pruning labels 
        # General track to PV, TODO need to check it, I think its about finding the PV which produce B
        yb = y_t_nodes[batch[('tracks', 'to', 'pvs')]['edge_index'][0]] * batch[('tracks', 'to', 'pvs')].y
        pv_sum = scatter_add(yb, batch[('tracks', 'to', 'pvs')].edge_index[1], dim=0)
        

        """Model output"""
        # Get the output w/o any pruning, later one w/ pruning for performance eval
        reco_batch = copy.deepcopy(batch)  # copying it for the reco peformance model w/ pruning
        outputs_np = self.model(batch)

        """Evaluate for the case whithout pruning, as pruning changes edge weights and output edges, nodes"""
        acc_LCA = acc_four_class(outputs_np[('tracks', 'to', 'tracks')].edges, y_tt_LCA)
        acc_tPV_edge = torch.sum(y_tPV_edges == (self.model._blocks[-1].edge_weights[('tracks', 'to', 'pvs')] > 0.5)) / y_tPV_edges.shape[0]
        pv_target = 1. * (pv_sum > 0)
        acc_PV_has_B = torch.sum(pv_target == (self.model._blocks[-1].node_weights['pvs'] > 0.5)) / pv_target.shape[0]
        
        # Get the block wide output
        for i, block in enumerate(self.model._blocks): 
            if self.get_node_performance:
                sig_nodes_selbool = y_t_nodes == 1
                sig_nodes_score = block.node_weights['tracks'].squeeze()[sig_nodes_selbool]
                bkg_nodes_score = block.node_weights['tracks'].squeeze()[~sig_nodes_selbool]
                self.tst_log[f"sig_nodes_score_{i}"] = torch.cat([self.tst_log[f"sig_nodes_score_{i}"], sig_nodes_score.cpu()], dim=0)
                self.tst_log[f"bkg_nodes_score_{i}"] = torch.cat([self.tst_log[f"bkg_nodes_score_{i}"], bkg_nodes_score.cpu()], dim=0)
            if self.get_edge_performance:
                sig_edges_selbool = y_tt_edges == 1
                sig_edges_score = block.edge_weights[('tracks', 'to', 'tracks')].squeeze()[sig_edges_selbool]
                bkg_edges_score = block.edge_weights[('tracks', 'to', 'tracks')].squeeze()[~sig_edges_selbool]
                self.tst_log[f"sig_edges_score_{i}"] = torch.cat([self.tst_log[f"sig_edges_score_{i}"], sig_edges_score.cpu()], dim=0)
                self.tst_log[f"bkg_edges_score_{i}"] = torch.cat([self.tst_log[f"bkg_edges_score_{i}"], bkg_edges_score.cpu()], dim=0)
            if self.get_frag_performance:
                true_frag_selbool = y_frag == 1
                frag_pos_part_score = torch.sigmoid(block.node_logits['frag'][true_frag_selbool]) # change to node weights
                frag_neg_part_score = torch.sigmoid(block.node_logits['frag'][~true_frag_selbool])
                self.tst_log[f"frag_pos_part_score_{i}"] = torch.cat([self.tst_log[f"frag_pos_part_score_{i}"], frag_pos_part_score.cpu()], dim=0)
                self.tst_log[f"frag_neg_part_score_{i}"] = torch.cat([self.tst_log[f"frag_neg_part_score_{i}"], frag_neg_part_score.cpu()], dim=0)
        
        # removing the output of the unprunded model
        del outputs_np

        """Start the study for reco performance"""
        # Activate the pruning for reco performance studies, needs to grab it form somewhere
        self.model._blocks[3].node_prune = True
        self.model._blocks[3].edge_prune = True
        outputs = self.model(reco_batch)
        self.model._blocks[3].node_prune = False
        self.model._blocks[3].edge_prune = False

        """Edge node prediciont plots output"""
        for i, block in enumerate(self.model._blocks):  # check if .item() is necessary
            # Obtain FT accuracy/block ouput
            if self.get_reco_performance:  # for confusion matrix
                bbar_ft_selbool = y_ft == 0
                none_ft_selbool = y_ft == 1
                b_ft_selbool = y_ft == 2

                ft_score = torch.softmax(block.node_logits['ft'], dim=1)
                bbar_ft_score =  torch.softmax(block.node_logits['ft'][bbar_ft_selbool], dim=1)
                none_ft_score = torch.softmax(block.node_logits['ft'][none_ft_selbool], dim=1)
                b_ft_score = torch.softmax(block.node_logits['ft'][b_ft_selbool], dim=1)

                self.tst_log[f"ft_score_{i}"] = torch.cat([self.tst_log[f"ft_score_{i}"], ft_score.cpu()], dim=0)
                self.tst_log[f"bbar_ft_score_{i}"] = torch.cat([self.tst_log[f"bbar_ft_score_{i}"], bbar_ft_score.cpu()], dim=0)
                self.tst_log[f"none_ft_score_{i}"] = torch.cat([self.tst_log[f"none_ft_score_{i}"], none_ft_score.cpu()], dim=0)
                self.tst_log[f"b_ft_score_{i}"] = torch.cat([self.tst_log[f"b_ft_score_{i}"], b_ft_score.cpu()], dim=0)

        """B reconstruction effiency"""
        if self.get_reco_performance:
            self.signal_df, self.event_df = eval_reco_performance(outputs, batch, batch_idx, self.signal_df, self.event_df, ft_score, self.ref_signal)

        """Logging"""
        for key, values in acc_LCA.items():
            self.tst_log[key].append(values)
        self.tst_log["tPV_edge_acc"].append(acc_tPV_edge)
        self.tst_log["PV_has_B_acc"].append(acc_PV_has_B)


    def on_test_epoch_start(self):
        self.init_tst_log()
        self.model._blocks[3].node_weight_cut = 0.01
        self.model._blocks[3].prune_by_cut = True
        self.model._blocks[3].edge_weight_cut = 0.01


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
    
    def on_test_epoch_end(self):
        if self.get_reco_performance:
            plot_ft_nodes(self.tst_log, len(self.model._blocks), self.version)
            self.signal_df.to_csv(f'lightning_logs/version_{self.version}/signal_df.csv', index=False)
            self.event_df.to_csv(f'lightning_logs/version_{self.version}/event_df.csv', index=False)
            del self.signal_df, self.event_df
        if self.get_node_performance:
            plot_gn_block_dist(self.tst_log, "nodes", len(self.model._blocks), self.version)
        if self.get_edge_performance:
            plot_gn_block_dist(self.tst_log, "edges", len(self.model._blocks), self.version)
        if self.get_frag_performance:
            plot_gn_block_dist(self.tst_log, "frag", len(self.model._blocks), self.version)


# Here is a trainer wrapper
def training(model, pos_weight, epochs, n_gpu, trn_loader, val_loader, accumulate_grad_batches=2, checkpoint_path=None):
    if checkpoint_path is None:
        module = HGNNLightningModule(
            model=model,
            pos_weights=pos_weight,
            optimizer_class=torch.optim.Adam,
            optimizer_params={"lr": 1e-3}
        )
    else:
        module = HGNNLightningModule.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        pos_weights=pos_weight,
        optimizer_class=torch.optim.Adam,
        optimizer_params={"lr": 1e-3}
        )

    early_stopping = EarlyStopping(
        monitor="val/combined_loss",
        verbose=True,
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

    log_dir = "lightning_logs"
    experiment_name = None  # default name

    tb_logger = TensorBoardLogger(save_dir=log_dir, name=experiment_name)
    csv_logger = CSVLogger(save_dir=log_dir, name=experiment_name, version=tb_logger.version)

    trainer = Trainer(
        logger=[csv_logger, tb_logger],
        max_epochs=epochs,
        accelerator='gpu',
        devices=n_gpu,
        strategy="auto",  # ddp_notebook change it to normal ddp don't do ddp_notebook takes way to much memory
        callbacks=[early_stopping, best_model_callback, all_epochs_callback],
        precision="32",  # never do 16-mixed
        gradient_clip_val=1.0,
        accumulate_grad_batches=accumulate_grad_batches,
        num_sanity_val_steps=1
    )

    """Start training"""
    trainer.fit(module, trn_loader, val_loader)