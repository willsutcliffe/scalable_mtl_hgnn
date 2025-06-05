# from IPython.core.completer import not_found

from wmpgnn.performance.reconstruction import reconstruct_decay, make_decay_dict
from wmpgnn.performance.reconstruction import particle_name, flatten, match_decays
from wmpgnn.util.functions import init_plot_style
from wmpgnn.util.functions import acc_four_class
from wmpgnn.model.model_loader import ModelLoader
from wmpgnn.lightning.lightning_module import HGNNLightningModule
from wmpgnn.datasets.data_handler import DataHandler
import pandas as pd
import numpy as np
import torch
import time
from torch_scatter import scatter_add
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


from itertools import chain
import os
import sys
import glob
from torch_geometric.loader import DataLoader

class Performance:
    """
    A class to evaluate the performance of a Graph Neural Network (GNN) model
    on various tasks, including Link Classification Accuracy (LCA),
    Primary Vertex (PV) association, track pruning, and decay chain reconstruction.

    It handles loading data, the pre-trained model, and provides methods
    for evaluating different aspects of the model's performance on test data.

    Attributes:
        config_loader : object
            The configuration object passed during initialization.
        data_loader : wmpgnn.datasets.data_handler.DataHandler
            An instance of `DataHandler` responsible for loading and managing datasets.
        full_graphs : bool
            Indicates whether full graphs are being processed.
        data_type : str
            The type of graph data ("homogeneous" or "heterogeneous") as specified
            in the configuration.
        dataset : torch.utils.data.DataLoader
            A PyTorch DataLoader for the test dataset.
        model : torch.nn.Module
            The loaded GNN model (`GNN` or `HeteroGNN`).
        name : str
            A name for the inference run, typically used for saving results.
        results_dir : str
            The directory where performance plots and results will be saved.
        cuda : bool
            Indicates if CUDA is being used.
        roc_auc_list : list
            A list to store ROC AUC scores, populated after calling `plot_roc_curve`.
        signal_df : pandas.DataFrame
            DataFrame to store detailed signal reconstruction performance metrics.
        event_df : pandas.DataFrame
            DataFrame to store detailed event-level performance metrics.

    """
    def __init__(self, config, full_graphs=False, cuda=True):
        """
                Initializes the Performance evaluation class, setting up data loaders,
                loading the pre-trained GNN model, and configuring plotting styles.

                Args:
                    config : object
                        A configuration object (e.g., from `config_loader`) that provides
                        parameters for data loading, model architecture, and inference.
                        It is expected to have a `get()` method to retrieve values.
                    full_graphs : bool, optional
                        If True, indicates that full graphs are being used. This might influence
                        how data is handled or interpreted by the model. Defaults to False.
                    cuda : bool, optional
                        If True, the model and data will be moved to a CUDA-enabled GPU for
                        computation, if available. Defaults to True.

                """
        self.config_loader = config

        model_loader = ModelLoader(self.config_loader)
        model = model_loader.get_model()

        pos_weight = {'t_nodes': torch.tensor(11.3371),'tt_edges': torch.tensor(480.6347), 'LCA': torch.tensor([2.5026e-01, 4.8915e+02, 7.4080e+02, 7.5415e+03]) }

        module = HGNNLightningModule.load_from_checkpoint(
                checkpoint_path="/eos/user/y/yukaiz/SWAN_projects/weighted_MP_gnn/wmpgnn//lightning/lightning_logs/version_3/checkpoints/epoch-epoch=05.ckpt",
                model=model,
                pos_weights=pos_weight,
                optimizer_class=torch.optim.Adam,
                optimizer_params={"lr": 1e-3})
        self.model = module.model
        self.model.eval()

        indir = "/eos/user/y/yukaiz/DFEI_data/Bs_JpsiPhi"
        tst_paths = sorted(glob.glob(f'{indir}/testing_data_*'))
        tst_dataset = list(chain.from_iterable(torch.load(p, weights_only=False) for p in tst_paths))

        self.data_loader = DataLoader(tst_dataset, batch_size=12, num_workers=1, drop_last=True)

        self.full_graphs = full_graphs
        self.data_type = config.get("dataset.data_type")
        self.dataset = DataLoader(tst_dataset, batch_size=8, num_workers=1, drop_last=True)

        self.full_graphs = full_graphs
        self.name = config.get("inference.name")
        self.results_dir = config.get("inference.results_dir")
        self.cuda = cuda
        if cuda:
            self.model.cuda()
        plt.rcParams.update(init_plot_style())




    def evaluate_hetero_lca_accuracy(self, prune_layer=3, bdt_pruned_data=False, batch_size=8):
        """
        Evaluates the Link Classification Accuracy (LCA) for a heterogeneous GNN model.

        This method calculates the accuracy of edge classifications, potentially
        considering pruning applied at a specific GNN layer.

        Args:
            prune_layer : int, optional
                The index of the GNN block (layer) where edge pruning might be applied.
                Defaults to 3.
            bdt_pruned_data : bool, optional
                If True, assumes that the data was pre-pruned by a BDT (Boosted Decision Tree)
                and concatenates original labels/edges with the current model's outputs
                for accuracy calculation. Defaults to False.
            batch_size : int, optional
                The batch size to use for the test DataLoader. Defaults to 8.

        Returns:
            torch.Tensor
                A tensor containing the mean accuracy across all batches,
                typically a 4-element tensor representing accuracy for each class
                or a combined accuracy. Returns `nanmean` to handle potential NaN values.
        """
        self.dataset = self.data_loader.get_test_dataloader(batch_size=batch_size)
        acc_one_epoch = []

        for i, data in enumerate(self.dataset):
            data.to('cuda')
            label0 = data[('tracks', 'to', 'tracks')].y.argmax(dim=1)
            y0 = data[('tracks', 'to', 'tracks')].y
            answers = torch.ones_like(data[('tracks', 'to', 'tracks')].edges).cuda()
            outputs = self.model(data)
            label = data[('tracks', 'to', 'tracks')].y.argmax(dim=1)

            if self.model._blocks[prune_layer].edge_prune == True:
                indices = self.model._blocks[prune_layer].edge_indices[('tracks', 'to', 'tracks')]
                index = torch.ones(label0.shape[0], dtype=bool).cuda()
                index[indices] = False
                selected_labels = label0[index]
                answers[:, 1:] = 0
                answers[indices] = outputs[('tracks', 'to', 'tracks')].edges

            if self.model._blocks[prune_layer].edge_prune == True:
                if bdt_pruned_data:
                    y_full = torch.concat([data.old_y, y0])
                    bdt_pruned_edges = torch.ones_like(data.old_y).cuda()
                    bdt_pruned_edges[:, 1:] = 0
                    edge_full = torch.concat([bdt_pruned_edges, answers])
                    acc_one_batch = acc_four_class(edge_full, y_full.argmax(dim=1))
                else:
                    acc_one_batch = acc_four_class(answers, label0)
            else:
                if bdt_pruned_data:
                    y_full = torch.concat([data.old_y, data[('tracks', 'to', 'tracks')].y])
                    bdt_pruned_edges = torch.ones_like(data.old_y).cuda()
                    bdt_pruned_edges[:, 1:] = 0
                    edge_full = torch.concat([bdt_pruned_edges, outputs[('tracks', 'to', 'tracks')].edges])
                    acc_one_batch = acc_four_class(edge_full, y_full.argmax(dim=1))
                else:
                    acc_one_batch = acc_four_class(outputs[('tracks', 'to', 'tracks')].edges, label)
            acc_one_epoch.append(acc_one_batch)
        acc_one_epoch = torch.stack(acc_one_epoch)
        return acc_one_epoch.nanmean(dim=0)

    def evaluate_homog_lca_accuracy(self, prune_layer=3, bdt_pruned_data=False, batch_size=8):
        """
        Evaluates the Link Classification Accuracy (LCA) for a homogeneous GNN model.

        This method calculates the accuracy of edge classifications, potentially
        considering pruning applied at a specific GNN layer for homogeneous graphs.

        Args:
            prune_layer : int, optional
                The index of the GNN block (layer) where edge pruning might be applied.
                Defaults to 3.
            bdt_pruned_data : bool, optional
                If True, assumes that the data was pre-pruned by a BDT and concatenates
                original labels/edges with the current model's outputs for accuracy calculation.
                Defaults to False.
            batch_size : int, optional
                The batch size to use for the test DataLoader. Defaults to 8.

        Returns:
            torch.Tensor
                A tensor containing the mean accuracy across all batches,
                typically a 4-element tensor representing accuracy for each class
                or a combined accuracy. Returns `nanmean` to handle potential NaN values.
        """
        self.dataset = self.data_loader.get_test_dataloader(batch_size=batch_size)
        acc_one_epoch = []


        for j, vdata in enumerate(self.dataset):
            vdata['graph_globals'] = vdata['graph_globals'].unsqueeze(1)
            vdata.receivers = vdata.receivers - torch.min(vdata.receivers)
            vdata.senders = vdata.senders - torch.min(vdata.senders)
            vdata.edgepos = vdata.edgepos - torch.min(vdata.edgepos)
            vdata.to('cuda')
            y0 = vdata.y
            label0 = vdata.y.argmax(dim=1)
            answers = torch.ones_like(vdata.edges).cuda()

            outputs = self.model(vdata)
            vdata = outputs
            label = vdata.y.argmax(dim=1)

            if self.model._blocks[prune_layer]._network.edge_prune == True:
                indices = self.model._blocks[prune_layer]._network.edge_indices
                index = torch.ones(label0.shape[0], dtype=bool).cuda()
                index[indices] = False
                selected_labels = label0[index]
                answers[:, 1:] = 0
                answers[indices] = outputs.edges

            if self.model._blocks[prune_layer]._network.edge_prune == True:
                if bdt_pruned_data:
                    y_full = torch.concat([vdata.old_y, y0])
                    bdt_pruned_edges = torch.ones_like(vdata.old_y).cuda()
                    bdt_pruned_edges[:, 1:] = 0
                    edge_full = torch.concat([bdt_pruned_edges, answers])
                    acc_one_batch = acc_four_class(edge_full, y_full.argmax(dim=1))
                else:
                    acc_one_batch = acc_four_class(answers, label0)
            else:
                if bdt_pruned_data:
                    y_full = torch.concat([vdata.old_y, vdata.y])
                    bdt_pruned_edges = torch.ones_like(vdata.old_y).cuda()
                    bdt_pruned_edges[:, 1:] = 0
                    edge_full = torch.concat([bdt_pruned_edges, outputs.edges])
                    acc_one_batch = acc_four_class(edge_full, y_full.argmax(dim=1))
                else:
                    acc_one_batch = acc_four_class(outputs.edges, label)
            acc_one_epoch.append(acc_one_batch)

        acc_one_epoch = torch.stack(acc_one_epoch)

        return acc_one_epoch.nanmean(dim=0)

    def set_edge_pruning(self, layer, cut, device='cuda'):
        """
        Enables and configures edge pruning for a specific GNN layer.

        Args:
            layer : int
                The index of the GNN block (layer) to apply edge pruning to.
            cut : float
                The threshold value for pruning edges. Edges with weights below this
                cut might be pruned.
            device : str, optional
                The device on which pruning operations should be performed ('cuda' or 'cpu').
                Defaults to 'cuda'.
        """
        if self.data_type == "homogeneous":
            self.model._blocks[layer]._network.edge_prune = True
            self.model._blocks[layer]._network.edge_weight_cut = cut
            self.model._blocks[layer]._network.prune_by_cut = True
            self.model._blocks[layer]._network.device = device
        elif self.data_type == "heterogeneous":
            self.model._blocks[layer].edge_prune = True
            self.model._blocks[layer].edge_weight_cut = cut
            self.model._blocks[layer].prune_by_cut = True
            self.model._blocks[layer].device = device

    def unset_pruning(self, layer):
        """
        Disables both edge and node pruning for a specific GNN layer.

        Args:
            layer : int
                The index of the GNN block (layer) to disable pruning for.
        """
        if self.data_type == "homogeneous":
            self.model._blocks[layer]._network.edge_prune = False
            self.model._blocks[layer]._network.node_prune = False
        elif self.data_type == "heterogeneous":
            self.model._blocks[layer].edge_prune = False
            self.model._blocks[layer].node_prune = False


    def set_node_pruning(self, layer, cut, device='cuda'):
        """
        Enables and configures node pruning for a specific GNN layer.

        Args:
            layer : int
                The index of the GNN block (layer) to apply node pruning to.
            cut : float
                The threshold value for pruning nodes. Nodes with weights below this
                cut might be pruned.
            device : str, optional
                The device on which pruning operations should be performed ('cuda' or 'cpu').
                Defaults to 'cuda'.
        """
        if self.data_type == "homogeneous":
            self.model._blocks[layer]._network.node_prune = True
            self.model._blocks[layer]._network.node_weight_cut = cut
            self.model._blocks[layer]._network.prune_by_cut = True
            self.model._blocks[layer]._network.device = device
        elif self.data_type == "heterogeneous":
            self.model._blocks[layer].node_prune = True
            self.model._blocks[layer].node_weight_cut = cut
            self.model._blocks[layer].prune_by_cut = True
            self.model._blocks[layer].device = device

    def set_pruning(self, layer, cut, device='cuda'):
        """
        Enables and configures both edge and node pruning for a specific GNN layer.

        This is a convenience method that calls `set_edge_pruning` and `set_node_pruning`.

        Args:
            layer : int
                The index of the GNN block (layer) to apply pruning to.
            cut : float
                The threshold value for pruning.
            device : str, optional
                The device on which pruning operations should be performed ('cuda' or 'cpu').
                Defaults to 'cuda'.
        """
        self.set_edge_pruning(layer, cut, device=device)
        self.set_node_pruning(layer, cut, device=device)

    def lca_reco_matrix(self, graph):
        """
        Generates a pandas DataFrame representing the reconstructed Link Classification
        Analysis (LCA) results from a GNN output graph.

        The DataFrame contains sender and receiver node indices, predicted LCA probabilities,
        and the predicted LCA decision (argmax of probabilities). It also filters
        for unique edge pairs (sender < receiver).

        Args:
            graph : torch_geometric.data.Data or torch_geometric.data.HeteroData
                The output graph object from the GNN model, containing edge information
                and predicted LCA probabilities.

        Returns:
            pandas.DataFrame
                A DataFrame with columns: 'senders', 'receivers', 'LCA_probs' (list of probabilities),
                and 'LCA_dec' (predicted class index).
        """
        if self.data_type == "homogeneous":
            senders = graph.senders
            receivers = graph.receivers
            edges = graph["edges"]
        elif self.data_type == "heterogeneous":
            senders = graph[('tracks', 'to', 'tracks')].edge_index[0]
            receivers = graph[('tracks', 'to', 'tracks')].edge_index[1]
            edges = graph[('tracks', 'to', 'tracks')].edges
        edge_index = torch.vstack([senders, receivers])
        pd_matrix = pd.DataFrame(np.vstack(
            (edge_index[0], edge_index[1])).transpose(), columns=['senders', 'receivers'])
        pd_matrix["LCA_probs"] = list(edges.detach().numpy())


        pd_matrix["LCA_dec"] = list(np.argmax(edges.detach().numpy(), axis=-1))
        pd_matrix.set_index(['senders', 'receivers'], inplace=True)
        pd_matrix = pd_matrix.reset_index()
        pd_matrix = pd_matrix[pd_matrix['senders'] < pd_matrix['receivers']]
        reverse_order_indices = list(map(tuple, np.vstack((graph["receivers"], graph["senders"])).transpose()))
        return pd_matrix

    def lca_truth_matrix(self, graph):
        """
        Generates a pandas DataFrame representing the ground truth Link Classification
        Analysis (LCA) for a given graph.

        This DataFrame includes true sender and receiver indices, true LCA decisions,
        particle names based on mother IDs, and the full LCA chain truth.

        Args:
            graph : torch_geometric.data.Data or torch_geometric.data.HeteroData
                The input graph object containing ground truth edge and particle information.

        Returns:
            pandas.DataFrame
                A DataFrame with columns: 'senders', 'receivers', 'LCA_dec' (true class index),
                'LCA_id_label' (particle name from mother ID), and 'TrueFullChainLCA' (boolean).
        """
        if self.data_type == "homogeneous":
            senders = graph.truth_senders
            receivers = graph.truth_receivers
            init_y = graph["truth_y"]
        elif self.data_type == "heterogeneous":
            senders = graph.truth_senders
            receivers = graph.truth_receivers
            init_y = graph["truth_y"]
        truth_lca = pd.DataFrame(np.column_stack((senders, receivers)), columns=['senders', 'receivers'])
        truth_lca['LCA_dec'] = np.reshape(
            np.argmax(
                np.reshape(init_y, (init_y.shape[0], 4)), axis=-1),
            (-1,))
        truth_lca = truth_lca[truth_lca['senders'] < truth_lca['receivers']]
        truth_lca['LCA_id_label'] = list(map(particle_name, graph['truth_moth_ids'].numpy()))
        truth_lca['TrueFullChainLCA'] = graph['lca_chain']
        return truth_lca

    def init_reco_dataframes(self):
        """
        Initializes the pandas DataFrames used for storing reconstruction performance
        metrics at both signal and event levels.

        This method sets up `self.signal_df` and `self.event_df` with predefined
        columns and data types.
        """
        self.signal_df = pd.DataFrame(
            columns=['EventNumber', 'NumParticlesInEvent', 'NumSignalParticles', 'PerfectSignalReconstruction',
                     'AllParticles', 'PerfectReco',
                     'NoneIso', 'PartReco', 'NotFound', 'SigMatch'])
        self.signal_df = self.signal_df.astype(
            {'EventNumber': np.int32, 'NumParticlesInEvent': np.int32, 'NumSignalParticles': np.int32,
             'PerfectSignalReconstruction': np.int32,
             'AllParticles': np.int32, 'PerfectReco': np.int32,
             'NoneIso': np.int32, 'PartReco': np.int32, 'NotFound': np.int32})

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

    def evaluate_pv_association(self, batch_size=8, b_tracks=True):
        """
        Evaluates the Primary Vertex (PV) association accuracy of the model.

        This method assesses how well the model associates tracks with their
        correct primary vertices.

        Args:
            batch_size : int, optional
                The batch size to use for the test DataLoader. Defaults to 8.
            b_tracks : bool, optional
                If True, only considers tracks associated with B-mesons (derived from
                `data[('tracks', 'to', 'tracks')].y[:, 0] == 0`). If False, considers all tracks.
                Defaults to True.

        Returns:
            tuple
                A tuple containing:
                - acc : float
                    The average accuracy of PV association across all processed batches.
                - npvs : list
                    A list of the number of primary vertices in each processed event.
                - associated : list
                    A list of integers (0 or 1) indicating whether each unique track
                    was correctly associated with its PV.
        """
        self.dataset = self.data_loader.get_test_dataloader(batch_size=batch_size)
        running_acc = 0
        npvs = []
        associated = []
        for i, data in enumerate(self.dataset):
            data.to('cuda')

            outputs = self.model(data)
            data = outputs
            PVlabel = torch.tensor(data[('tracks', 'to', 'pvs')].y, dtype=torch.float32)

            if b_tracks:
                tracks = data[('tracks', 'to', 'tracks')].edge_index[0][data[('tracks', 'to', 'tracks')].y[:, 0] == 0]
            else:
                tracks = data[('tracks', 'to', 'tracks')].edge_index[0]
            unique_tracks = torch.unique(tracks)
            correctly_associated = 0

            for i in unique_tracks:
                index = (data[('tracks', 'to', 'pvs')].edge_index[0] == i)
                pv_associated = (torch.argmax(
                    self.model._blocks[-1].edge_weights[('tracks', 'to', 'pvs')][index]) == torch.argmax(
                    data[('tracks', 'to', 'pvs')].y[index]))
                correctly_associated += int(pv_associated.item())
                npvs.append(data['pvs'].x.shape[0])
                associated.append( int(pv_associated.item()))
            running_acc += correctly_associated / unique_tracks.shape[0]
        acc = running_acc / len(self.dataset)
        return acc, npvs, associated

    def evaluate_homog_track_pruning_performance(self, layers=[0, 1, 2, 7], batch_size=8,
                                                 edge_pruning=True, plot_roc=False,
                                                  show_plots = False):
        """
        Evaluates the track (edge or node) pruning performance for a homogeneous GNN model.

        This method calculates and optionally plots histograms of weights and ROC curves
        to assess how effectively the model prunes tracks based on edge or node weights.

        Parameters:
            layers : list of int, optional
                A list of GNN layer indices for which to evaluate pruning performance.
                Defaults to [0, 1, 2, 7].
            batch_size : int, optional
                The batch size to use for the test DataLoader. Defaults to 8.
            edge_pruning : bool, optional
                If True, evaluates edge pruning; otherwise, evaluates node pruning.
                Defaults to True.
            plot_roc : bool, optional
                If True, generates and plots ROC curves for the pruning performance.
                Defaults to False.

        Returns:
            tuple
                A tuple containing:
                - true : numpy.ndarray
                    The concatenated ground truth labels (0 or 1) for edges/nodes.
                - pred : list of numpy.ndarray
                    A list of concatenated predicted weights for each specified layer.
        """
        trues = []
        self.dataset = self.data_loader.get_test_dataloader(batch_size=batch_size)

        preds = {}
        for layer in layers:
            preds[layer] = []
        for j, data in enumerate(self.dataset):

            data['graph_globals'] = data['graph_globals'].unsqueeze(1)
            data.receivers = data.receivers - torch.min(data.receivers)
            data.senders = data.senders - torch.min(data.senders)
            data.edgepos = data.edgepos - torch.min(data.edgepos)
            data.to('cuda')
            outputs = self.model(data)

            label = data.y.argmax(dim=1)
            num_nodes = data.nodes.shape[0]
            out = data.edges.new_zeros(num_nodes, data.edges.shape[1])
            node_sum = scatter_add(data.y, data.senders, out=out, dim=0)
            ynodes = (1. * (torch.sum(node_sum[:, 1:], 1) > 0)).unsqueeze(1)
            if edge_pruning:
                trues.append(1 * (outputs.cpu().y.detach().numpy()[:, 0] == 0))
                for layer in layers:
                    preds[layer].append(self.model._blocks[layer]._network.edge_weights.cpu().detach().numpy())
            else:
                trues.append(ynodes.cpu().detach().numpy())
                for layer in layers:
                    preds[layer].append(self.model._blocks[layer]._network.node_weights.cpu().detach().numpy())
        true = np.concatenate(trues)
        pred = [np.concatenate(preds[i]) for i in layers]
        names = [f"Layer {i}" for i in layers]

        if edge_pruning:
            identifier = "edge"
        else:
            identifier = "node"
        for i in range(len(pred)):
            plt.hist(pred[i][true==1], bins=100,density=True, label="y=1", histtype="step")
            plt.hist(pred[i][true==0], bins=100, density=True, label="y=0", histtype="step")
            plt.legend()
            plt.xlabel("Weights")
            plt.savefig(f"{self.results_dir}/{self.name}_{identifier}_{names[i]}_histogram.png", dpi=300)
            plt.savefig(f"{self.results_dir}/{self.name}_{identifier}_{names[i]}_histogram.pdf", dpi=300)
            if show_plots:
                plt.show()
        if plot_roc:
            if edge_pruning:
                self.plot_roc_curve(true, pred, names, file_name="hetero_edge_pruning_roc", title="Edge")
            else:
                self.plot_roc_curve(true, pred, names, file_name="hetero_node_pruning_roc", title="Node")

        return true, pred

    def evaluate_hetero_track_pruning_performance(self, layers=[0, 1, 2, 7], batch_size=8,
                                                  edge_pruning=True, plot_roc=False, pv_tr_edges = False,
                                                  show_plots = False):
        """
        Evaluates the track (edge or node) pruning performance for a heterogeneous GNN model.

        This method calculates and optionally plots histograms of weights and ROC curves
        to assess how effectively the model prunes tracks based on edge or node weights
        in a heterogeneous graph context. It can also evaluate pruning for PV-track edges.

        Args:
            layers : list of int, optional
                A list of GNN layer indices for which to evaluate pruning performance.
                Defaults to [0, 1, 2, 7].
            batch_size : int, optional
                The batch size to use for the test DataLoader. Defaults to 8.
            edge_pruning : bool, optional
                If True, evaluates edge pruning; otherwise, evaluates node pruning.
                Defaults to True.
            plot_roc : bool, optional
                If True, generates and plots ROC curves for the pruning performance.
                Defaults to False.
            pv_tr_edges : bool, optional
                If True, evaluates pruning performance specifically for 'tracks' to 'pvs'
                (Primary Vertex) edges. Defaults to False.

        Returns:
            tuple
                A tuple containing:
                - true : numpy.ndarray
                    The concatenated ground truth labels (0 or 1) for edges/nodes.
                - pred : list of numpy.ndarray
                    A list of concatenated predicted weights for each specified layer.
        """
        trues = []
        self.dataset = self.data_loader.get_test_dataloader(batch_size=batch_size)

        preds = {}
        for layer in layers:
            preds[layer] = []
        for i, data in enumerate(self.dataset):
            data.to('cuda')
            outputs = self.model(data)
            data = outputs
            label = data[('tracks', 'to', 'tracks')].y.argmax(dim=1)
            PVlabel = torch.tensor(data[('tracks', 'to', 'pvs')].y, dtype=torch.float32)
            num_nodes = data['tracks'].x.shape[0]
            out = data[('tracks', 'to', 'tracks')].edges.new_zeros(num_nodes,
                                                                   data[('tracks', 'to', 'tracks')].y.shape[1])
            node_sum = scatter_add(data[('tracks', 'to', 'tracks')].y, data[('tracks', 'to', 'tracks')].edge_index[0],
                                   out=out, dim=0)
            ynodes = (1. * (torch.sum(node_sum[:, 1:], 1) > 0)).unsqueeze(1)
            yBCE = 1. * (data[('tracks', 'to', 'tracks')].y[:, 0] == 0).unsqueeze(1)

            if edge_pruning:
                if pv_tr_edges:
                    trues.append(PVlabel.cpu().detach().numpy())
                else:
                    trues.append(yBCE.cpu().detach().numpy())
                for layer in layers:
                    if pv_tr_edges:
                        preds[layer].append(
                        self.model._blocks[layer].edge_weights[('tracks', 'to', 'pvs')].cpu().detach().numpy())
                    else:
                        preds[layer].append(
                        self.model._blocks[layer].edge_weights[('tracks', 'to', 'tracks')].cpu().detach().numpy())
            else:
                trues.append(ynodes.cpu().detach().numpy())
                for layer in layers:
                    preds[layer].append(
                        self.model._blocks[layer].node_weights['tracks'].cpu().detach().numpy())
        true = np.concatenate(trues)
        pred = [np.concatenate(preds[i]) for i in layers]
        names = [f"Layer {i}" for i in layers]
        if edge_pruning:
            identifier = "edge"
            if pv_tr_edges:
                identifier = "pv_edge"
        else:
            identifier = "node"
        for i in range(len(pred)):
            plt.hist(pred[i][true==1], bins=100,density=True, label="y=1", histtype="step")
            plt.hist(pred[i][true==0], bins=100, density=True, label="y=0", histtype="step")
            plt.legend()
            plt.xlabel("Weights")
            plt.savefig(f"{self.results_dir}/{identifier}_{names[i]}_histogram.png", dpi=300)
            plt.savefig(f"{self.results_dir}/{identifier}_{names[i]}_histogram.pdf", dpi=300)
            if show_plots:
                plt.show()
        if plot_roc:
            if edge_pruning:
                if pv_tr_edges:
                    self.plot_roc_curve(true, pred, names, file_name="hetero_pv_edge_pruning_roc", title="PV Edge", show_plots=show_plots)
                else:
                    self.plot_roc_curve(true, pred, names, file_name="hetero_edge_pruning_roc", title="Edge", show_plots=show_plots)
            else:
                self.plot_roc_curve(true, pred, names, file_name="hetero_node_pruning_roc", title="Node", show_plots=show_plots)
        return true, pred

    def plot_roc_curve(self, y_true, y_scores, names, file_name="test_edge_pruning_roc", title="Edge", show_plots=False):
        """
        Generates and plots ROC (Receiver Operating Characteristic) curves for pruning performance.

        This method takes true labels and predicted scores (weights) for multiple scenarios
        (e.g., different layers) and plots their corresponding ROC curves,
        calculating and displaying the Area Under the Curve (AUC) for each.

        Args:
            y_true : numpy.ndarray
                Ground truth binary labels (0 or 1).
            y_scores : list of numpy.ndarray
                A list where each element is an array of predicted scores (weights)
                corresponding to `y_true`. Each array represents a different curve to plot.
            names : list of str
                A list of names for each ROC curve, corresponding to `y_scores`, used
                in the legend.
            file_name : str, optional
                The base name for the output image files (PNG and PDF).
                Defaults to "test_edge_pruning_roc".
            title : str, optional
                The title for the plot. Defaults to "Edge".

        """
        plt.figure(figsize=(8, 6))

        line_styles = ["dashed", "dotted","dashdot", "solid",(0, (5, 10))]
        colors = ['black','red', 'brown', 'blue', 'purple']
        self.roc_auc_list = []
        for i in range(0, len(y_scores)):
            name = names[i]
            y_score = y_scores[i]
            line_style = line_styles[i]
            color = colors[i]
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_score = roc_auc_score(y_true, y_score)
            self.roc_auc_list.append(auc_score)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.4f})', linestyle=line_style, color=color)
            print(auc_score)
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel('Background retention')
        plt.ylabel('Signal efficiency')
        plt.title(f'{title} pruning ')
        plt.legend()
        plt.grid()
        plt.savefig(f"{self.results_dir}/{self.name}_{file_name}.png", dpi=300)
        plt.savefig(f"{self.results_dir}/{self.name}_{file_name}.pdf")
        if show_plots:
            plt.show()

    def evaluate_reco_performance(self, event_max=-1, plot_perfect_decaychains=2,
                                  pruning_cut=0, ref_signal = None):
        self.dataset = self.data_loader.get_test_dataloader(batch_size=1)
        """
            Evaluates the full reconstruction performance of the model, including decay chain
            reconstruction and various signal/event-level metrics.

            This method iterates through the test dataset, performs inference, reconstructs
            decay chains, and populates internal DataFrames (`signal_df`, `event_df`)
            with detailed performance statistics. It can optionally plot perfectly
            reconstructed decay chains.

            Args:
                event_max : int, optional
                    The maximum number of events to process for evaluation. If -1, all
                    events in the dataset are processed. Defaults to -1.
                plot_perfect_decaychains : int, optional
                    The number of perfectly reconstructed decay chains to plot and save.
                    If 0, no plots are generated. Defaults to 2.
                pruning_cut : float, optional
                    A pruning cut value to apply to the model's output (specifically for
                    layer 7, if applicable) during inference. If 0, no pruning is applied.
                    Defaults to 0.
                ref_signal : list of dict, optional
                    A reference signal definition, typically a list of dictionaries. Each dictionary
                    should contain 'mothers' and 'daughters' keys, with lists of particle names.
                    This is used for matching reconstructed decays to a specific signal
                    (e.g., specific B-meson decay channels). Defaults to None.

            Returns
            -------
            None
                This method does not return any value. Instead, it populates
                `self.signal_df` and `self.event_df` with performance metrics and
                saves them as CSV files. It also calls `self.performance_table`
                to generate a LaTeX table summary.
            """
        self.init_reco_dataframes()
        time_node_filtering = 0
        time_edge_filtering = 0
        time_LCA_reconstruction = 0
        event = 0
        time_model = 0
        time_reco = 0
        if self.cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
        for batch_i, vdata in enumerate(self.dataset):
            if batch_i == event_max:
                print(f"Event loop for reconstruction ending at event max {event_max}")
                break
            if event % 100 == 0:
                print(event)
            Bparts = float(torch.sum(torch.argmax(vdata.truth_y, -1) > 0))
            if Bparts < 1:
                continue
            if self.data_type == "homogeneous":
                vdata['graph_globals'] = vdata['graph_globals'].unsqueeze(1)
                vdata.receivers = vdata.receivers - torch.min(vdata.receivers)
                vdata.senders = vdata.senders - torch.min(vdata.senders)
                vdata.edgepos = vdata.edgepos - torch.min(vdata.edgepos)

            if self.cuda:
                vdata.cuda()
                self.model.cuda()
                if pruning_cut > 0:
                    self.set_pruning(7, pruning_cut)
            else:
                if pruning_cut > 0:
                    self.set_pruning(7, pruning_cut, device="cpu")

            if self.cuda:
                torch.cuda.synchronize()
                start.record()
                with torch.no_grad():
                    gout = self.model(vdata)
                end.record()
                torch.cuda.synchronize()
                time_model = start.elapsed_time(end)
            else:
                start_time = time.time()
                with torch.no_grad():
                    gout = self.model(vdata)
                end_time = time.time()
                time_model = end_time - start_time

            if self.cuda:
                #self.model.cpu()
                gout.cpu()
                vdata.cpu()
            if self.data_type == "homogeneous":
                Bparts_after = float(torch.sum(torch.argmax(vdata.y, -1) > 0))
            elif self.data_type == "heterogeneous":
                Bparts_after = float(torch.sum(torch.argmax(vdata[('tracks', 'to', 'tracks')].y, -1) > 0))
            #if Bparts_after < Bparts:
            #    print("Bparts lost, before had ", Bparts, " and now ", Bparts_after)

            start_time = time.time()
            reco_LCA = self.lca_reco_matrix(gout)
            particle_keys = list(vdata["final_keys"].numpy())
            total_number_of_particles = len(particle_keys)
            reco_cluster_dict, reco_num_clusters_per_order, _ = reconstruct_decay(
                reco_LCA, particle_keys)
            end_time = time.time()
            time_reco = end_time - start_time
            start_time = time.time()
            true_LCA = self.lca_truth_matrix(vdata)

            particle_keys = list(vdata["truth_part_keys"].numpy())
            particle_ids = list(map(particle_name, vdata['truth_part_ids'].numpy()))
            truth_cluster_dict, truth_num_clusters_per_order, max_full_chain_depth_in_event = reconstruct_decay(
                true_LCA, particle_keys, particle_ids=particle_ids, truth_level_simulation=1)
            end_time = time.time()
            time_truth = end_time - start_time
            time_node_filtering = 0
            time_edge_filtering = 0
            time_LCA_reconstruction = time_reco

            if truth_cluster_dict != {}:
                # Compute performance metrics
                particles_from_heavy_hadron = flatten(
                    [truth_cluster_dict[tc_firstkey]['node_keys'] for tc_firstkey in truth_cluster_dict.keys()])
                number_of_particles_from_heavy_hadron = len(particles_from_heavy_hadron)

                number_of_background_particles = total_number_of_particles - number_of_particles_from_heavy_hadron
                if reco_cluster_dict != {}:
                    selected_particles = flatten(
                        [reco_cluster_dict[tc_firstkey]['node_keys'] for tc_firstkey in reco_cluster_dict.keys()])
                    number_of_selected_particles = len(selected_particles)
                    number_of_selected_particles_from_heavy_hadron = len(
                        list(set(selected_particles).intersection(particles_from_heavy_hadron)))
                    number_of_selected_background_particles = number_of_selected_particles - number_of_selected_particles_from_heavy_hadron
                else:
                    number_of_selected_particles = 0
                    number_of_selected_particles_from_heavy_hadron = 0
                    number_of_selected_background_particles = 0

                perfect_event_reconstruction = 1
                if number_of_selected_background_particles > 0:
                    perfect_event_reconstruction = 0

                for tc_firstkey in truth_cluster_dict.keys():
                    signal_match = 1

                    if ref_signal != None:
                        labels = truth_cluster_dict[tc_firstkey]['labels']
                        mothers = [label[3:] for label in labels if 'c' == label[0]]
                        node_keys  = truth_cluster_dict[tc_firstkey]['node_keys']
                        daughters =  [label.split(':')[1] for label in labels if int(label.split(':')[0][1:]) in node_keys]

                        if match_decays(daughters, ref_signal[0]['daughters']) or match_decays(daughters, ref_signal[1]['daughters']):
                            signal_match = 1
                        else:
                            signal_match = 0

                        if signal_match == 1:
                            check_mothers1 = True
                            check_mothers2 = True
                            for i in range(len(ref_signal[0]['mothers'])):
                                if ref_signal[0]['mothers'][i] not in mothers:
                                    check_mothers1 = False
                                if ref_signal[1]['mothers'][i] not in mothers:
                                    check_mothers2 = False
                            if check_mothers1 or check_mothers2:
                                signal_match = 1
                            else:
                                signal_match = 0


                    number_of_signal_particles = len(truth_cluster_dict[tc_firstkey]['node_keys'])

                    perfect_signal_reconstruction = 1
                    if reco_cluster_dict == {}:
                        perfect_signal_reconstruction = 0
                    else:
                        if tc_firstkey not in reco_cluster_dict.keys():
                            perfect_signal_reconstruction = 0
                        else:
                            if reco_cluster_dict[tc_firstkey]['node_keys'] != truth_cluster_dict[tc_firstkey][
                                'node_keys'] or reco_cluster_dict[tc_firstkey]['LCA_values'] != \
                                    truth_cluster_dict[tc_firstkey]['LCA_values']:
                                perfect_signal_reconstruction = 0
                    perfect_event_reconstruction *= perfect_signal_reconstruction

                    true_cluster = truth_cluster_dict[tc_firstkey]
                    perfect_reco = 0
                    all_particles = 0
                    none_iso = 0
                    part_reco = 0
                    none_associated = 0
                    reco = []
                    for cluster in reco_cluster_dict.values():
                        reco.append(cluster['node_keys'])
                        true_in_reco = np.sum(np.isin(true_cluster['node_keys'], cluster['node_keys'])) / len(
                            true_cluster['node_keys'])

                        if cluster['node_keys'] == true_cluster['node_keys']:
                            all_particles = 1
                            if cluster['LCA_values'] == true_cluster['LCA_values']:
                                perfect_reco = 1
                            break
                        elif true_in_reco == 1 and len(cluster['node_keys']) > len(true_cluster['node_keys']):
                            none_iso = 1
                            break
                        elif true_in_reco >= 0.2 and true_in_reco < 1:
                            part_reco = 1

                    if all_particles == 1:
                        none_iso = 0
                        part_reco = 0
                    if none_iso == 1:
                        part_reco = 0

                    if all_particles == 0 and none_iso == 0 and part_reco == 0:
                        none_associated = 1



                    self.signal_df = self.signal_df._append({'EventNumber': event,
                                                             'NumParticlesInEvent': total_number_of_particles,
                                                             'NumSignalParticles': number_of_signal_particles,
                                                             'PerfectSignalReconstruction': perfect_signal_reconstruction,
                                                             'AllParticles': all_particles,
                                                             'PerfectReco': perfect_reco,
                                                             'NoneIso': none_iso,
                                                             'PartReco': part_reco,
                                                             'NotFound': none_associated,
                                                             'SigMatch': signal_match},
                                                            ignore_index=True)
                    if perfect_signal_reconstruction and plot_perfect_decaychains > 0:
                        plt.clf()
                        fix, axs = plt.subplots(2, figsize=(10, 10))
                        axs[0].set_title('Reco trees in event',
                                         fontweight='bold', fontsize=14)
                        particle_keys = list(vdata["final_keys"].numpy())
                        reco_cluster_dict, reco_num_clusters_per_order, _ = reconstruct_decay(
                            reco_LCA, particle_keys, axs[0])
                        axs[1].set_title('Truth-level trees in event',
                                         fontweight='bold', fontsize=14)

                        particle_keys = list(vdata["truth_part_keys"].numpy())
                        particle_ids = list(map(particle_name, vdata['truth_part_ids'].numpy()))
                        truth_cluster_dict, truth_num_clusters_per_order, max_full_chain_depth_in_event = reconstruct_decay(
                            true_LCA, particle_keys, axs[1], particle_ids=particle_ids, truth_level_simulation=1)
                        plt.savefig(f"{self.results_dir}/{self.name}_perfect_reco_decay_chain_{plot_perfect_decaychains}.png",dpi=300)
                        plt.savefig(f"{self.results_dir}/{self.name}_perfect_reco_decay_chain_{plot_perfect_decaychains}.pdf")
                        plot_perfect_decaychains = plot_perfect_decaychains - 1

                event += 1
                if number_of_background_particles <=0:
                    number_of_background_particles = -1
                if number_of_particles_from_heavy_hadron <=0:
                    number_of_particles_from_heavy_hadron = -1
                self.event_df = self.event_df._append({'EventNumber': event,

                                                       'NumParticlesInEvent': total_number_of_particles,
                                                       'NumParticlesFromHeavyHadronInEvent': number_of_particles_from_heavy_hadron,
                                                       'NumBackgroundParticlesInEvent': number_of_background_particles,
                                                       'NumSelectedParticlesInEvent': number_of_selected_particles,
                                                       'NumSelectedParticlesFromHeavyHadronInEvent': number_of_selected_particles_from_heavy_hadron,

                                                       'NumSelectedBackgroundParticlesInEvent': number_of_selected_background_particles,
                                                       'NumTruthClustersGen1': truth_num_clusters_per_order[0],
                                                       'NumTruthClustersGen2': truth_num_clusters_per_order[1],
                                                       'NumTruthClustersGen3': truth_num_clusters_per_order[2],
                                                       'NumTruthClustersGen4': truth_num_clusters_per_order[3],
                                                       'NumRecoClustersGen1': reco_num_clusters_per_order[0],
                                                       'NumRecoClustersGen2': reco_num_clusters_per_order[1],
                                                       'NumRecoClustersGen3': reco_num_clusters_per_order[2],
                                                       'NumRecoClustersGen4': reco_num_clusters_per_order[3],
                                                       'MaxTruthFullChainDepthInEvent': max_full_chain_depth_in_event,
                                                       'EfficiencyParticlesFromHeavyHadronInEvent': float(
                                                           number_of_selected_particles_from_heavy_hadron) / number_of_particles_from_heavy_hadron,
                                                       'EfficiencyBackgroundParticlesInEvent': float(
                                                           number_of_selected_background_particles) / number_of_background_particles,
                                                       'BackgroundRejectionPowerInEvent': 1. - float(
                                                           number_of_selected_background_particles) / number_of_background_particles,
                                                       'PerfectEventReconstruction': perfect_event_reconstruction,
                                                       'TimeNodeFiltering': time_node_filtering,
                                                       'TimeEdgeFiltering': time_edge_filtering,
                                                       'TimeLCAReconstruction': time_LCA_reconstruction,
                                                       'TimeSequence': time_node_filtering + time_edge_filtering + time_LCA_reconstruction,
                                                       'NumTrueSignalsInEvent': len(truth_cluster_dict.keys()),
                                                       'NumRecoSignalsInEvent': len(reco_cluster_dict.keys()),
                                                       'TimeModel': time_model,
                                                       'TimeReco': time_reco,
                                                       'TimeTruth': time_truth},
                                                      ignore_index=True)

        if ref_signal != None:
            self.performance_table(signal_match=True)
        else:
            self.performance_table()
        self.signal_df.to_csv(f"{self.results_dir}/{self.name}_signal_df.csv")
        self.event_df.to_csv(f"{self.results_dir}/{self.name}_event_df.csv")


    def performance_table(self, signal_match=False):
        """
        Generates and prints a LaTeX-formatted performance table summarizing
        the decay chain reconstruction results at both signal and event levels.

        The table includes metrics such as "Perfect hierarchy", "Wrong hierarchy",
        "None isolated", and "Part reco" (partially reconstructed/not found).
        These metrics are calculated as percentages based on the data in
        `self.signal_df` and `self.event_df`.

        Args:
            signal_match : bool, optional
                If True, only signal instances that match a predefined reference signal
                (as determined during `evaluate_reco_performance` if `ref_signal` was provided)
                are considered for the "True b" scope. If False, all signal instances
                are included. Defaults to False.

        Returns:
            None
                This method does not return any value. It prints the LaTeX table to
                the console and saves it to a file named
                `{self.results_dir}/{self.name}_performance_table.tex`.
        """
        perf_numbers = pd.DataFrame(columns=["Scope", "Perfect hierarchy", "Wrong hierarchy", "None isolated", "Part reco"])

        if signal_match:
            signal_df = self.signal_df.query("SigMatch == 1")
        else:
            signal_df = self.signal_df
        sig_perfect_reco = 100 * len(signal_df.query('PerfectSignalReconstruction == 1')) / len(signal_df)
        sig_wrong_hierarchy = 100 * len(signal_df.query('AllParticles == 1')) / len(signal_df) - sig_perfect_reco
        sig_none_isolated = 100 * len(signal_df.query('NoneIso == 1')) / len(signal_df)
        sig_part_reco = 100 * (len(signal_df.query('PartReco == 1')) / len(signal_df) + len(
            signal_df.query('NotFound == 1')) / len(signal_df))

        event_perfect_reco = 100 * len(self.event_df.query('PerfectEventReconstruction == 1')) / len(self.event_df)
        event_wrong_hierarchy = 100 * len(self.event_df.query(
            'NumSelectedBackgroundParticlesInEvent == 0 and NumSelectedParticlesFromHeavyHadronInEvent 	==  NumParticlesFromHeavyHadronInEvent')) / len(
            self.event_df) - event_perfect_reco
        event_none_isolated = 100 * len(self.event_df.query(
            'NumSelectedParticlesFromHeavyHadronInEvent ==  NumParticlesFromHeavyHadronInEvent and NumSelectedBackgroundParticlesInEvent > 0')) / len(
            self.event_df)
        event_part_reco = 100 * len(
            self.event_df.query('NumSelectedParticlesFromHeavyHadronInEvent <  NumParticlesFromHeavyHadronInEvent')) / len(
            self.event_df)
        perf_numbers = perf_numbers._append({"Scope": "True b", "Perfect hierarchy": sig_perfect_reco,
                                             "Wrong hierarchy": sig_wrong_hierarchy, "None isolated": sig_none_isolated,
                                             "Part reco": sig_part_reco}, ignore_index=True)
        perf_numbers = perf_numbers._append({"Scope": "Event", "Perfect hierarchy": event_perfect_reco,
                                             "Wrong hierarchy": event_wrong_hierarchy, "None isolated": event_none_isolated,
                                             "Part reco": event_part_reco},
                                            ignore_index=True)
        with open(f"{self.results_dir}/{self.name}_performance_table.tex", "w") as f:
            f.write(perf_numbers.to_latex(index=False, float_format="{:.1f}".format))
            print(perf_numbers.to_latex(index=False, float_format="{:.1f}".format))


    def heterogeneous_performance(self):
        """
        Evaluates the performance of the GNN model on heterogeneous graph data.

        This method assesses various aspects of the model's performance specific
        to heterogeneous graphs, including Link Classification Accuracy (LCA)
        with and without edge pruning, track pruning performance (both edge
        and node-based), and Primary Vertex (PV) association accuracy for
        different track types.

        It prints intermediate results to the console.

        Returns:
            None
                This method does not return any value. It performs evaluations
                and prints the results directly. The `roc_auc_list` attribute
                is populated internally during track pruning evaluations.
        """
        val_acc_no_pruning = None
        if self.full_graphs:
            val_acc_no_pruning = self.evaluate_hetero_lca_accuracy(bdt_pruned_data=False)
        else:
            val_acc_no_pruning = self.evaluate_hetero_lca_accuracy(bdt_pruned_data=True)
        print(val_acc_no_pruning)
        self.set_edge_pruning(3, 0.001)
        val_acc_pruning = self.evaluate_hetero_lca_accuracy(bdt_pruned_data=True)
        self.unset_edge_pruning(3)
        print(val_acc_pruning)
        self.evaluate_hetero_track_pruning_performance(layers=[0,1,2,7] ,edge_pruning=True)
        edge_roc_scores = self.roc_auc_list
        self.evaluate_hetero_track_pruning_performance(layers=[0,1,2,7] ,edge_pruning=False)
        node_roc_scores = self.roc_auc_list

        b_track_pv_association, _, _ = self.evaluate_pv_association(b_tracks=True)
        all_track_pv_association, _, _ = self.evaluate_pv_association(b_tracks=False)
        print("b track ", b_track_pv_association)
        print("all track ", all_track_pv_association)



    def homogeneous_performance(self):
        """
        Evaluates the performance of the GNN model on homogeneous graph data.

        This method assesses various aspects of the model's performance specific
        to homogeneous graphs, including Link Classification Accuracy (LCA)
        with and without edge pruning, and track pruning performance (both edge
        and node-based).

        It prints intermediate results to the console and compiles key performance
        metrics into a Pandas DataFrame.

        Returns:
            None
                This method does not return any value. It performs evaluations,
                prints the results, and creates a Pandas DataFrame named `performance`
                containing summarized metrics, though this DataFrame is not explicitly
                returned or saved within this method. The `roc_auc_list` attribute
                is populated internally during track pruning evaluations.
        """
        val_acc_no_pruning = None
        if self.full_graphs:
            val_acc_no_pruning = self.evaluate_homog_lca_accuracy(bdt_pruned_data=False)
        else:
            val_acc_no_pruning = self.evaluate_homog_lca_accuracy(bdt_pruned_data=True)
        print(val_acc_no_pruning)
        self.set_edge_pruning(3, 0.001)
        val_acc_pruning = self.evaluate_homog_lca_accuracy(bdt_pruned_data=True)
        self.unset_edge_pruning(3)
        print(val_acc_pruning)
        self.evaluate_homog_track_pruning_performance(layers=[0,1,2,7] ,edge_pruning=True)
        edge_roc_scores = self.roc_auc_list
        self.evaluate_homog_track_pruning_performance(layers=[0,1,2,7] ,edge_pruning=False)
        node_roc_scores = self.roc_auc_list

        performance = pd.DataFrame( data = {
            'before_pruning_lca0' : [val_acc_no_pruning[0]],
            'before_pruning_lca1': [val_acc_no_pruning[1]],
            'before_pruning_lca2': [val_acc_no_pruning[2]],
            'before_pruning_lca3': [val_acc_no_pruning[3]],
            'after_pruning_lca0': [val_acc_pruning[0]],
            'after_pruning_lca1': [val_acc_pruning[1]],
            'after_pruning_lca2': [val_acc_pruning[2]],
            'after_pruning_lca3': [val_acc_pruning[3]],
            'edge_layer_0_auc': [edge_roc_scores[0]],
            'edge_layer_1_auc': [edge_roc_scores[1]],
            'edge_layer_2_auc': [edge_roc_scores[2]],
            'edge_layer_7_auc': [edge_roc_scores[3]],
            'node_layer_0_auc': [node_roc_scores[0]],
            'node_layer_1_auc': [node_roc_scores[1]],
            'node_layer_2_auc': [node_roc_scores[2]],
            'node_layer_7_auc': [node_roc_scores[3]]
        } )

