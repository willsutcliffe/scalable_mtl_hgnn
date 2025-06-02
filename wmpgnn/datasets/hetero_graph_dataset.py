import torch
import numpy as np
from torch_geometric.data import Dataset, Data
from torch_geometric.data import HeteroData


def find_row_indices(t1, t2):
    """
    Find matching row indices between two tensors and return one-hot encoded results.

    This function finds which rows in t2 match each row in t1, returning both the
    indices and a one-hot encoded representation. Used for matching tracks to
    reconstructed primary vertices.

    Args:
        t1 (torch.Tensor): First tensor to match (e.g., true PV coordinates)
        t2 (torch.Tensor): Second tensor to match against (e.g., reconstructed PVs)

    Returns:
        tuple: A tuple containing:
            - indices (torch.Tensor): Index of matching row in t2 for each row in t1,
                                    or -1 if no match found
            - one_hot_encoded (torch.Tensor): One-hot encoded representation of the matches
    """
    t1_expanded = t1.unsqueeze(1)
    t2_expanded = t2.unsqueeze(0)

    matches = (t1_expanded == t2_expanded).all(dim=2)

    matches_int = matches.int()

    indices = torch.argmax(matches_int, dim=1)

    valid = matches.any(dim=1)
    indices[~valid] = -1


    num_classes = t2.shape[0]

    one_hot_encoded = torch.nn.functional.one_hot(indices, num_classes=num_classes)

    return indices, one_hot_encoded

class CustomHeteroDataset(Dataset):
    """
    A custom PyTorch dataset for loading heterogeneous graph data for particle physics applications.

    This dataset processes particle tracking data by creating heterogeneous graphs with multiple
    node types (tracks, primary vertices, globals) and edge types. It handles track-to-track
    relationships and track-to-primary-vertex associations, computing impact parameters and
    creating bipartite connections between tracks and reconstructed primary vertices.

    The dataset is designed for particle physics vertex reconstruction tasks using heterogeneous
    Graph Neural Networks with PyTorch Geometric.

    Attributes:
        filenames_input (list): List of file paths for input graph data
        filenames_target (list): List of file paths for target/label data
        performance_mode (bool): Whether to include additional performance evaluation data
    """
    def __init__(self, filenames_input, filenames_target, performance_mode=False):
        """
        Initialize the CustomHeteroDataset.

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
        Load and process all heterogeneous graph data from files.

        This method creates heterogeneous graphs with three node types:
        - 'tracks': Particle tracks with position, momentum, and additional features
        - 'pvs': Reconstructed primary vertices
        - 'globals': Global event-level features

        And two edge types:
        - ('tracks', 'pvs'): Bipartite connections between tracks and primary vertices
        - ('tracks', 'tracks'): Track-to-track relationships

        The method computes impact parameters for track-PV associations and creates
        one-hot encoded labels for track-PV matching.

        Returns:
            list: List of PyTorch Geometric HeteroData objects containing:

            Node types and features:
                - data['tracks'].x: Track features [position(3), momentum(3), additional(1)]
                - data['pvs'].x: Primary vertex coordinates [x, y, z]
                - data['globals'].x: Global features + number of PVs

            Edge types and features:
                - data['tracks', 'pvs'].edge_index: Bipartite track-PV connections
                - data['tracks', 'pvs'].y: One-hot encoded track-PV associations
                - data['tracks', 'pvs'].edges: Impact parameters between tracks and PVs
                - data['tracks', 'tracks'].edge_index: Track-to-track connections
                - data['tracks', 'tracks'].y: Track-track edge labels
                - data['tracks', 'tracks'].edges: Track-track edge features

        Additional fields when performance_mode=True:
                - final_keys: Node key mappings
                - moth_ids, part_ids: Particle identification
                - lca_chain: Lowest common ancestor chain
                - truth_*: Ground truth connectivity and labels
                - true_origin: True origin information
                - truth_reco_pv: True reconstructed primary vertex coordinates

        Note:
            - Performs node remapping for contiguous track indexing
            - Computes impact parameters using 3D distance formula
            - Handles NaN values in impact parameter calculations
            - Creates Cartesian product for all track-PV combinations
            - Extracts unique reconstructed primary vertices from track data
        """
        data_set = []
        C = 0
        j = 0
        for i in range(self.__len__()):
            graph = np.load(self.filenames_input[i], allow_pickle=True).item()
            graph_target = np.load(self.filenames_target[i], allow_pickle=True).item()
            labels = np.array(graph_target["edges"])
            indices = np.unique(graph['receivers'])
            remapping = {a: i for a, i in zip(indices, list(range(0, len(indices))))}
            old_senders = graph["senders"]
            old_receivers = graph["receivers"]
            senders = np.array([remapping[x] for x in graph["senders"]])
            receivers = np.array([remapping[x] for x in graph["receivers"]])
            senders = torch.from_numpy(senders).long()
            receivers = torch.from_numpy(receivers).long()
            new_nodes = graph["nodes"][indices]
            new_edges = graph['edges']
            new_nodes = torch.from_numpy(new_nodes)
            new_edges = torch.from_numpy(new_edges)


            recoPVs = torch.unique(new_nodes[:, -3:], dim=0)
            nPVs = recoPVs.shape[0]

            true_nodes_PVs = new_nodes[:, -3:]

            y, y_one_hot = find_row_indices(true_nodes_PVs, recoPVs)

            xyz = new_nodes[:, :3]
            P = new_nodes[:, 3:6]

            xyz_repeated = xyz.unsqueeze(1).repeat(1, recoPVs.shape[0], 1)
            P_repeated = P.unsqueeze(1).repeat(1, recoPVs.shape[0], 1)
            r = xyz_repeated - recoPVs
            IPs = torch.sqrt(
                torch.sum(r ** 2, dim=-1) - torch.sum(P_repeated * r, dim=-1) ** 2 / torch.sum(P_repeated ** 2, dim=-1))

            if torch.isnan(IPs).any().item():
                IPs[torch.isnan(IPs)] = torch.max(IPs[~torch.isnan(IPs)]).item()

            permutations = torch.cartesian_prod(torch.arange(true_nodes_PVs.shape[0]), torch.arange(recoPVs.shape[0]))
            data = HeteroData()


            truth_reco_pv = new_nodes[:, -3:]
            new_nodes = torch.hstack([new_nodes[:, :6], new_nodes[:, 9:10]])

            data['tracks'].x = new_nodes

            data['pvs'].x = recoPVs
            data['globals'].x = torch.hstack(
                [torch.from_numpy(graph["globals"]), torch.tensor(nPVs, dtype=torch.float32)]).unsqueeze(0)


            data['tracks', 'pvs'].edge_index = permutations.T
            data['tracks', 'pvs'].y = y_one_hot.flatten().unsqueeze(-1)
            data['tracks', 'pvs'].edges = IPs.flatten().unsqueeze(-1)

            data['tracks', 'tracks'].edge_index = torch.vstack([senders, receivers])
            data['tracks', 'tracks'].y = torch.from_numpy(labels)
            data['tracks', 'tracks'].edges = new_edges

            if self.performance_mode:
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
                data['true_origin'] = torch.from_numpy(graph["nodes"][indices][:, 13:])
                data['truth_reco_pv'] = truth_reco_pv


            data_set.append(data)

        return data_set