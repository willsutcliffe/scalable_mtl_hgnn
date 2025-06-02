import torch
import numpy as np
from torch_geometric.data import Dataset, Data


class CustomDataset(Dataset):
    """
    A custom PyTorch dataset for loading and processing graph data from files.

    This dataset loads graph data from numpy files, processes node and edge information,
    performs node remapping, and optionally includes performance mode data for evaluation.
    The dataset is designed for graph neural network training with PyTorch Geometric.

    Attributes:
        filenames_input (list): List of file paths for input graph data
        filenames_target (list): List of file paths for target/label data
        performance_mode (bool): Whether to include additional performance evaluation data
    """
    def __init__(self, filenames_input, filenames_target, performance_mode=False):
        """
        Initialize the CustomDataset.

        Args:
            filenames_input (list): List of file paths to input graph data (.npy files)
            filenames_target (list): List of file paths to target/label data (.npy files)
            performance_mode (bool, optional): If True, includes additional data for
                                             performance evaluation and analysis. Defaults to False.
        """
        self.filenames_input = filenames_input
        self.filenames_target = filenames_target
        self.performance_mode = performance_mode

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns:
            int: Number of target files (dataset size)
        """
        return len(self.filenames_target)

    def len(self):
        """
        Alternative method to get dataset length (PyTorch Geometric compatibility).

        Returns:
            int: Number of target files (dataset size)
        """
        return len(self.filenames_target)

    def update(self, **kwargs):
        """
        Update dataset attributes with provided keyword arguments.

        Args:
            **kwargs: Arbitrary keyword arguments to update instance attributes
        """
        self.__dict__.update(kwargs)

    def get(self):
        """
        Load and process all graph data from files.

        This method processes each graph file pair, performs node remapping to create
        contiguous indices, extracts relevant features, and creates PyTorch Geometric
        Data objects. Optionally includes performance evaluation data.

        Returns:
            list: List of PyTorch Geometric Data objects containing:
                - nodes: Node features (first 10 columns)
                - edges: Edge features
                - senders: Source node indices (remapped and offset)
                - receivers: Target node indices (remapped and offset)
                - graph_globals: Global graph features
                - edgepos: Edge position indices
                - y: Edge labels/targets
                - num_edges: Number of edges in the graph
                - num_nodes: Number of nodes in the graph
                - true_reco_pv: True reconstructed primary vertex (columns 10-13)

        Additional fields when performance_mode=True:
                - init_senders, init_receivers: Initial graph connectivity
                - init_y: Initial edge labels
                - init_keys, final_keys: Node key mappings
                - init_moth_ids, moth_ids: Mother particle IDs
                - init_partids, part_ids: Particle IDs
                - lca_chain: Lowest common ancestor chain
                - truth_*: Ground truth connectivity and labels
                - true_origin: True origin information (columns 13+)
                - old_y: Edges present in initial but not final graph

        Note:
            - Skips graphs with no senders or receivers
            - Performs node index remapping for contiguous indexing
            - Applies global offset (C) to maintain unique node indices across graphs
            - In performance mode, computes Cantor pairing for edge matching
        """
        data_set = []
        C = 0
        j = 0
        for i in range(self.__len__()):
            graph = np.load(self.filenames_input[i], allow_pickle=True).item()
            graph_target = np.load(self.filenames_target[i], allow_pickle=True).item()
            labels = graph_target["edges"]
            indices = np.unique(graph['receivers'])
            remapping = {a: i for a, i in zip(indices, list(range(0, len(indices))))}
            old_senders = graph["senders"]
            old_receivers = graph["receivers"]
            if len(graph["receivers"]) < 1 or len(graph["senders"]) < 1:
                continue
            s = np.array([remapping[x] for x in graph["senders"]])
            r = np.array([remapping[x] for x in graph["receivers"]])
            new_nodes = graph["nodes"][indices][:, : 10]
            true_reco_pv =  graph["nodes"][indices][:, 10:13]
            new_edges = graph['edges']
            data = Data(nodes=torch.from_numpy(new_nodes),
                        edges=torch.from_numpy(new_edges),
                        senders=torch.from_numpy(s + C).long(),
                        receivers=torch.from_numpy(r + C).long(),
                        graph_globals=torch.from_numpy(graph["globals"]),
                        edgepos=torch.from_numpy(np.array([j] * graph["edges"].shape[0])).long(),
                        y=torch.from_numpy(labels),
                        num_edges=torch.tensor(graph["edges"].shape[0]),
                        num_nodes=torch.tensor(graph["nodes"][indices].shape[0]),
                        true_reco_pv= torch.from_numpy(true_reco_pv)
                        )
            if self.performance_mode:
                data['init_senders'] = torch.from_numpy(graph["init_y"]["senders"]).long()
                data['init_receivers'] = torch.from_numpy(graph["init_y"]["receivers"]).long()
                data['init_y'] = torch.from_numpy(graph["init_y"]["edges"])
                data['init_keys'] = torch.from_numpy(graph["init_keys"])
                data['init_moth_ids'] = torch.from_numpy(graph["init_ids"])
                data['init_partids'] = torch.from_numpy(graph["init_part_ids"])
                data['final_keys'] = torch.from_numpy(graph["keys"])
                data['moth_ids'] = torch.from_numpy(graph["ids"])
                data['part_ids'] = torch.from_numpy(graph["part_ids"])
                data['lca_chain'] = torch.from_numpy(graph["lca_chain"])
                data['truth_senders'] = torch.from_numpy(graph["truth_senders"]).long()
                data['truth_receivers'] = torch.from_numpy(graph["truth_receivers"]).long()
                data['truth_y'] = torch.from_numpy(graph["truth_y"])
                data['truth_moth_ids'] = torch.from_numpy(graph["truth_ids"])
                data['truth_part_ids'] = torch.from_numpy(graph["truth_part_ids"])
                data['truth_part_keys'] = torch.from_numpy(graph["truth_part_keys"])
                data['truth_part_ids'] = torch.from_numpy(graph["truth_part_ids"])
                data['true_origin'] = torch.from_numpy(graph["nodes"][indices][:, 13:])
                init_senders = data.init_keys[data.init_senders]
                init_receivers = data.init_keys[data.init_receivers]
                senders =  data.final_keys[torch.from_numpy(old_senders).long()]
                receivers =  data.final_keys[torch.from_numpy(old_receivers).long()]
                init_cantor = 0.5 * (init_senders + init_receivers - 2) * (
                            init_senders + init_receivers - 1) + init_senders
                final_cantor = 0.5 * (senders + receivers - 2) * (senders + receivers - 1) + senders
                data['old_y'] = data.init_y[~torch.isin(init_cantor, final_cantor)]
            data_set.append(data)

            C += np.max(r) + 1
            j += 1


        return data_set
