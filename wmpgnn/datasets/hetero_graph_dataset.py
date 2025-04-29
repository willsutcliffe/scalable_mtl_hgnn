import torch
import numpy as np
from torch_geometric.data import Dataset, Data
from torch_geometric.data import HeteroData


def find_row_indices(t1, t2):
    # Expand t1 and t2 to compare every row in t1 with every row in t2
    t1_expanded = t1.unsqueeze(1)  # Shape (n1, 1, cols)
    t2_expanded = t2.unsqueeze(0)  # Shape (1, n2, cols)

    # Check for equality along the last dimension
    matches = (t1_expanded == t2_expanded).all(dim=2)  # Shape (n1, n2)

    # Convert matches to an appropriate type for argmax
    matches_int = matches.int()  # Convert to int

    # Find the indices in t2 for each row in t1
    indices = torch.argmax(matches_int, dim=1)


    # Verify matches exist (if no match, indices would be arbitrary)
    valid = matches.any(dim=1)
    indices[~valid] = -1  # Set unmatched rows to -1
    indice = torch.tensor([1, 0, 2])

    # Number of classes (largest value in the tensor + 1)
    num_classes = t2.shape[0]

    # One-hot encoding
    one_hot_encoded = torch.nn.functional.one_hot(indices, num_classes=num_classes)

    return indices, one_hot_encoded

class CustomHeteroDataset(Dataset):
    def __init__(self, filenames_input, filenames_target, performance_mode=False, n_classes=5):
        self.filenames_input = filenames_input
        self.filenames_target = filenames_target
        self.performance_mode = performance_mode
        self.n_classes = n_classes

    # No. of graphs
    def __len__(self):
        return len(self.filenames_target)

    def len(self):
        return len(self.filenames_target)

    def update(self, **kwargs):
        self.__dict__.update(kwargs)

    def get(self):
        data_set = []
        C = 0
        j = 0
        for i in range(self.__len__()):
            graph = np.load(self.filenames_input[i], allow_pickle=True).item()
            if graph['nodes'].shape[0]==0:
                print(f"Empty {i} graph: {self.filenames_input[i]}")
                continue
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

            # last three columns are the PVs XYZ
            recoPVs = torch.unique(new_nodes[:, -3:], dim=0)
            nPVs = recoPVs.shape[0]
            true_nodes_PVs = new_nodes[:, -3:]
            if nPVs == 0:
                import pdb; pdb.set_trace()
            # print(torch.sum(torch.sum(nodes_PVs == true_nodes_PVs,dim=-1)==3)/nodes_PVs.shape[0])
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
            new_nodes = new_nodes[:, :-6]
            data['tracks'].x = new_nodes

            data['pvs'].x = recoPVs
            data['globals'].x = torch.hstack(
                [torch.from_numpy(graph["globals"]), torch.tensor(nPVs, dtype=torch.float32)]).unsqueeze(0)


            data['tracks', 'pvs'].edge_index = permutations.T
            data['tracks', 'pvs'].y = y_one_hot.flatten().unsqueeze(-1)
            data['tracks', 'pvs'].edges = IPs.flatten().unsqueeze(-1)

            data['tracks', 'tracks'].edge_index = torch.vstack([senders, receivers])
            data['tracks', 'tracks'].y = torch.from_numpy(labels[:, :self.n_classes])
            data['tracks', 'tracks'].edges = new_edges

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
                init_senders = data.init_keys[data.init_senders]
                init_receivers = data.init_keys[data.init_receivers]
                senders =  data.final_keys[torch.from_numpy(old_senders).long()]
                receivers =  data.final_keys[torch.from_numpy(old_receivers).long()]
                init_cantor = 0.5 * (init_senders + init_receivers - 2) * (
                            init_senders + init_receivers - 1) + init_senders
                final_cantor = 0.5 * (senders + receivers - 2) * (senders + receivers - 1) + senders
                data['old_y'] = data.init_y[~torch.isin(init_cantor, final_cantor)]

            data_set.append(data)

        return data_set