import pytorch_lightning as L
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from collections import defaultdict
import copy

import torch

from lightning_module_helper import *

from wmpgnn.util.functions import acc_four_class
from wmpgnn.util.pruners import edge_pruning, true_node_pruning
from wmpgnn.performance.plotter import *
from wmpgnn.performance.tagging_power import analyze_tagging_power
from wmpgnn.performance.reconstruction import reco_event

torch.set_float32_matmul_precision("high")


class HGNNLightningModule(L.LightningModule):
    def __init__(self, model, optimizer_class, optimizer_params, config, pos_weights, is_train=True):
        super().__init__()
        self.automatic_optimization = False
        self.is_train = is_train
        if is_train:
            self.version = None
            self.save_hyperparameters({
                **config,
                "pos_weights": make_loggable(pos_weights)
            })
        else:
            self.version = config["training"]["cpt"]["model"].split("_")[0]

        self.signal = config["evaluate"]["sample"]
        self.dfei_usage = config["model"]["DFEI"]["usage"]
        self.ft_usage = config["model"]["FT_inferer"]["usage"]
        self.model = model

        # include here the second model, which only transforms the output
        self.config = config["training"]["infer"]
        self.nFT_layers = config["model"]["DFEI"]["GNblocks"]["FTlayers"]
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params

        # Loss functions
        if self.config["LCA"]:
            self.lca_criterion = nn.CrossEntropyLoss(weight=pos_weights["LCA"])
        if self.config["node_prune"]:
            self.node_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights["nodes"])
        if self.config["edge_prune"]:
            self.edge_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights["edges"])
        if self.config["frag"]:
            self.frag_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights["frag"])
        if self.config["FT"]:
            self.ft_criterion = nn.CrossEntropyLoss(weight=pos_weights["FT"])

        self.trn_log, self.val_log = init_logs(config)
        self.tst_log = init_logs(config, mode="test")
        self.sig_df, self.evt_df = None, None

        # Pruning threshold for reco
        self.edge_prune = 0.1
        self.node_prune = 0.1

    def forward(self, batch):
        return self.model(batch)

    def configure_optimizers(self):
        params = {}
        optimizers = []

        if hasattr(self.model, "dfei_model"):
            params["dfei_model"] = self.model.dfei_model.parameters()
            optimizers.append(self.optimizer_class(params["dfei_model"], **self.optimizer_params))

        if hasattr(self.model, "ft_model"):
            params["ft_model"] = self.model.ft_model.parameters()
            optimizers.append(self.optimizer_class(params["ft_model"], **self.optimizer_params))
        return optimizers

    def shared_step(self, batch, batch_idx, log, mode="train"):
        if mode != "test":
            optimizers = self.optimizers()
            if not isinstance(optimizers, (list, tuple)):
                optimizers = [optimizers]

        loss = init_loss(self.device)

        # Get y value of ft
        if self.config["FT"]:
            y_ft = batch['tracks'].ft
        else:
            ft_des = torch.ones(batch['tracks'].ft.shape[0], 3) * -1

        """First model pass"""
        if self.dfei_usage:
            data = copy.deepcopy(batch)
            outputs = self.model(data)

            if self.config["LCA"]:
                y_LCA = data[('tracks', 'to', 'tracks')].y.to(torch.int64)
                loss["LCA"] = self.lca_criterion(outputs[('tracks', 'to', 'tracks')].lca, y_LCA)
                log["LCA_loss"].append(loss["LCA"].item())
                acc_LCA = acc_four_class(outputs[('tracks', 'to', 'tracks')].lca, y_LCA)
                for key, values in acc_LCA.items():
                    log[key].append(values)

            if self.config["node_prune"]:
                if self.config["sim"] == "pythia":
                    y_nodes = (data["tracks"].ft != 1).to(torch.float32).unsqueeze(-1)
                else:
                    num_nodes = data['tracks'].x.shape[0]
                    out = data[('tracks', 'to', 'tracks')].edges.new_zeros(num_nodes,
                                                                           data[('tracks', 'to', 'tracks')].y.shape[
                                                                               1])
                    node_sum = scatter_add(data[('tracks', 'to', 'tracks')].y,
                                           data[('tracks', 'to', 'tracks')].edge_index[0],
                                           out=out, dim=0)
                    y_nodes = ((torch.sum(node_sum[:, 1:], 1) > 0)).unsqueeze(1).float()
            if self.config["edge_prune"]:
                y_edges = data[('tracks', 'to', 'tracks')].y > 0
                y_edges = y_edges.to(torch.float32).unsqueeze(-1)
            if self.config["frag"]:  # Frag does not work
                y_frag = (data['tracks'].frag != 0).unsqueeze(-1).to(torch.float32)

            for i, block in enumerate(self.model.dfei_model._blocks):
                if self.config["node_prune"]:
                    loss["t_nodes"] += self.node_criterion(block.node_logits['tracks'], y_nodes)
                    if mode == "test":
                        get_node_score(log, block.node_weights['tracks'].squeeze(), y_nodes, i)

                if self.config["edge_prune"]:
                    loss["tt_edges"] += self.edge_criterion(block.edge_logits[('tracks', 'to', 'tracks')], y_edges)
                    if mode == "test":
                        get_edge_score(log, block.edge_weights[('tracks', 'to', 'tracks')].squeeze(), y_edges, i)

                if i >= len(self.model.dfei_model._blocks) - self.nFT_layers:
                    if self.config["frag"]:
                        loss["frag_nodes"] += self.frag_criterion(block.node_logits['frag'], y_frag)
                    if self.config["FT"]:
                        selbool = y_ft != 1
                        loss["ft_nodes"] +=self.ft_criterion(block.node_logits['ft'][selbool], y_ft[selbool])

        # Combined loss of the FT model
        combined_loss = loss["LCA"] + loss["t_nodes"] + 33 * loss["tt_edges"] + loss["frag_nodes"] + loss["ft_nodes"]
        if mode == "train" and self.dfei_usage:
            optimizers[0].zero_grad()
            self.manual_backward(combined_loss, retain_graph=True)
            optimizers[0].step()

        if self.ft_usage and self.config["FT"]:
            # here the model is actually updated, by the first one
            outputs_ft = self.model(batch)

            selbool = y_ft != 1
            ift_loss = self.ft_criterion(outputs_ft["tracks"].x[selbool], y_ft[selbool])
            loss["ft_nodes"] += ift_loss
            combined_loss += ift_loss
            if mode == "test":
                ft_des = torch.softmax(outputs_ft["tracks"].x, dim=1)
            if mode == "train":
                optimizers[-1].zero_grad()
                self.manual_backward(ift_loss)
                optimizers[-1].step()

        if mode == "test":
            if self.dfei_usage:
                graph = outputs
            elif self.ft_usage:
                graph = outputs_ft

            frag_selbool = graph["tracks"].frag != -1
            frag_in_evt = graph["tracks"].frag[frag_selbool]
            frag_pid = graph["part_ids"][frag_selbool]

            if self.config["node_prune"] or self.config["edge_prune"]:
                node_selbool = block.node_weights["tracks"].squeeze() > self.node_prune
                edge_mask = true_node_pruning(node_selbool, graph, "tracks", [('tracks', 'to', 'tracks')])
                ft_des = ft_des[node_selbool]
                edge_selbool = block.edge_weights[('tracks', 'to', 'tracks')].squeeze()[edge_mask] > self.edge_prune
                edge_pruning(edge_selbool, graph, ('tracks', 'to', 'tracks'))
                graph[("tracks", "to", "tracks")].lca = graph[("tracks", "to", "tracks")].lca[edge_mask][edge_selbool]
            graph["frag_y"] = frag_in_evt
            graph["frag_pid"] = frag_pid
            self.sig_df, self.evt_df = reco_event(graph, batch_idx, self.config, self.signal, self.sig_df,
                                                  self.evt_df, ft_des)

        """Logging"""
        log = loss_logging(log, loss, self.config)
        log["combined_loss"].append(combined_loss.item())
        return combined_loss

    def on_test_start(self) -> None:
        self.sig_df, self.evt_df = init_test_df()

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, self.trn_log, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, self.val_log, mode="val")
        return loss

    def test_step(self, batch, batch_idx):
        _ = self.shared_step(batch, batch_idx, self.tst_log, mode="test")
        return {}

    def on_train_epoch_end(self):
        avg_losses = epoch_end_loggable(self.trn_log)
        for key, val in avg_losses.items():
            self.log(f"train_{key}", val, prog_bar=(key == "combined_loss"), on_epoch=True, on_step=False)
        self.trn_log = defaultdict(list)

    def on_validation_epoch_end(self):
        avg_losses = epoch_end_loggable(self.val_log)
        for key, val in avg_losses.items():
            self.log(f"val_{key}", val, prog_bar=(key == "combined_loss"), on_epoch=True, on_step=False)
        self.val_log = defaultdict(list)

    def on_test_epoch_end(self):
        if self.is_train:
            self.version = self.logger.version
        self.sig_df.to_csv(f'lightning_logs/version_{self.version}/signal_df_{self.signal}.csv', index=False)
        self.evt_df.to_csv(f'lightning_logs/version_{self.version}/event_df_{self.signal}.csv', index=False)
        if self.config["node_prune"]:
            for i in range(len(self.model.dfei_model._blocks)):
                plot_weights(self.tst_log[f"sig_nodes_score_{i}"], self.tst_log[f"bkg_nodes_score_{i}"],
                             [f"NN_nodes_{i}", "sig", "bkg"], self.version, channel=self.signal)
        if self.config["edge_prune"]:
            for i in range(len(self.model.dfei_model._blocks)):
                plot_weights(self.tst_log[f"sig_edges_score_{i}"], self.tst_log[f"bkg_edges_score_{i}"],
                             [f"NN_edges_{i}", "sig", "bkg"], self.version, channel=self.signal)
        if self.config["LCA"]:
            obtain_reco_accuracy(self.sig_df, self.version, self.signal)
        if self.config["FT"]:
            process_ft(self.tst_log, self.sig_df, self.version, self.signal)
            analyze_tagging_power(self.sig_df, self.version, self.signal)


# Here we define a wrapper to do the training
def training(model, trn_loader, val_loader, tst_loader, config, pos_weights):
    seed_everything(42, workers=True)
    load_from_cpt = config["training"]["cpt"]["model"]
    if load_from_cpt == "None":
        module = HGNNLightningModule(
            model=model,
            optimizer_class=torch.optim.Adam,
            optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
            config=config,
            pos_weights=pos_weights
        )
        first_batch = next(iter(trn_loader))
        with torch.no_grad():
            module(first_batch)
        print("initialized")
        print("=" * 30)
        del first_batch
    else:
        print("Loading from checkpoint")
        print(load_from_cpt)
        module = HGNNLightningModule.load_from_checkpoint(
            checkpoint_path=load_from_cpt,
            model=model,
            pos_weights=pos_weights,
            optimizer_class=torch.optim.Adam,
            optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
            config=config
        )
    module = torch.compile(module)

    early_stopping = EarlyStopping(
        monitor="val_combined_loss",
        verbose=True,
        mode="min",
        patience=10,
    )

    best_model_callback = ModelCheckpoint(
        filename="best-{epoch:02d}-{val_combined_loss:.2f}",
        monitor="val_combined_loss",
        mode="min",
        save_top_k=5
    )

    log_dir = "lightning_logs"
    experiment_name = None  # default name

    tb_logger = TensorBoardLogger(save_dir=log_dir, name=experiment_name)
    version = tb_logger.version
    csv_logger = CSVLogger(save_dir=log_dir, name=experiment_name, version=tb_logger.version)

    config = config["training"]
    trainer = Trainer(
        logger=[csv_logger, tb_logger],
        max_epochs=config["epochs"],
        accelerator="gpu",
        devices=config["ngpu"],
        strategy="auto",
        callbacks=[early_stopping, best_model_callback],
        precision="32",
        accumulate_grad_batches=config["gacc"],
        num_sanity_val_steps=0,
        sync_batchnorm=True,
        reload_dataloaders_every_n_epochs=1,
        deterministic=True
    )

    """Start training"""
    trainer.fit(module, trn_loader, val_loader)

    # testing
    run_test = any(value for key, value in config["infer"].items() if key != "LCA")
    if run_test:
        trainer.test(module, dataloaders=tst_loader)

    csv_path = os.path.join(log_dir, f"version_{version}", "metrics.csv")
    df = pd.read_csv(csv_path)
    df = df.groupby('epoch').agg(lambda x: x.dropna().iloc[0] if not x.dropna().empty else None).reset_index()
    return df, version


def evaluate(model, tst_loader, config, pos_weight):
    load_from_cpt = config["training"]["cpt"]["model"]

    if load_from_cpt == "None":
        raise RuntimeError("No model defined")
    else:
        print("Loading from checkpoint")
        cpt = get_model_path(load_from_cpt)[0]
        print(cpt)
        module = HGNNLightningModule.load_from_checkpoint(
            checkpoint_path=cpt,
            model=model,
            pos_weights=pos_weight,
            optimizer_class=torch.optim.Adam,
            optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
            scheduler_params={"min_lr": 1e-4, "patience": 5},
            config=config,
            is_train=False
        )
        module = torch.compile(module)
    version = config["training"]["cpt"]["model"].split("_")[0]
    trainer = Trainer(
        default_root_dir=f'lightning_logs/version_{version}',  # save the eval stuff in the dir of the model
    )

    trainer.test(module, dataloaders=tst_loader)
    csv_path = os.path.join("lightning_logs", f"version_{version}", "metrics.csv")
    df = pd.read_csv(csv_path)
    df = df.groupby('epoch').agg(lambda x: x.dropna().iloc[0] if not x.dropna().empty else None).reset_index()
    return df, version
