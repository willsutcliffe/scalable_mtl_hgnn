import torch
import numpy as np
from torch_geometric.data import Dataset, Data


class CustomDataset(Dataset):
    def __init__(self, filenames_input, filenames_target, performance_mode=False):
        self.filenames_input = filenames_input
        self.filenames_target = filenames_target
        self.performance_mode = performance_mode
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
            graph_target = np.load(self.filenames_target[i], allow_pickle=True).item()
            labels = graph_target["edges"]
            indices = np.unique(graph['receivers'])
            remapping = {a: i for a, i in zip(indices, list(range(0, len(indices))))}
            old_senders = graph["senders"]
            old_receivers = graph["receivers"]
            s = np.array([remapping[x] for x in graph["senders"]])
            r = np.array([remapping[x] for x in graph["receivers"]])
            # new_nodes = np.take(graph["nodes"][indices], [0, 1, 2, 3, 4, 5,9], axis=1)
            # new_edges = np.take(graph["edges"], [1, 2, 3], axis=1)
            new_nodes = graph["nodes"][indices][:, : 10]
            
            new_edges = graph['edges']
            data = Data(nodes=torch.from_numpy(new_nodes),  # node features
                        #             data = Data(nodes=torch.from_numpy(graph["nodes"][indices]), #node features

                        #                         edges=torch.from_numpy(graph["edges"]), #edge features
                        edges=torch.from_numpy(new_edges),  # edge features

                        senders=torch.from_numpy(s + C).long(),
                        receivers=torch.from_numpy(r + C).long(),
                        graph_globals=torch.from_numpy(graph["globals"]),
                        edgepos=torch.from_numpy(np.array([j] * graph["edges"].shape[0])).long(),
                        y=torch.from_numpy(labels),
                        num_edges=torch.tensor(graph["edges"].shape[0]),
                        num_nodes=torch.tensor(graph["nodes"][indices].shape[0])
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
                init_senders = data.init_keys[data.init_senders]
                init_receivers = data.init_keys[data.init_receivers]
                senders =  data.final_keys[torch.from_numpy(old_senders).long()]
                receivers =  data.final_keys[torch.from_numpy(old_receivers).long()]
                init_cantor = 0.5 * (init_senders + init_receivers - 2) * (
                            init_senders + init_receivers - 1) + init_senders
                final_cantor = 0.5 * (senders + receivers - 2) * (senders + receivers - 1) + senders
                data['old_y'] = data.init_y[~torch.isin(init_cantor, final_cantor)]
            data_set.append(data)
            # print(data['receivers'])
            # print(remapping)
            C += np.max(r) + 1
            # print(torch.tensor(np.array([j]*graph["edges"].shape[0])))
            j += 1
            # print(C)

        return data_set
