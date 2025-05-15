import torch
import numpy as np
import pandas as pd
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

class CustomNeutralsHeteroDataset(Dataset):
    def __init__(self, filenames_input, filenames_target, performance_mode=False, n_classes=2):
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
        for i in range(len(self.filenames_input)):
            graph = np.load(self.filenames_input[i], allow_pickle=True).item()
            if graph['nodes'].shape[0] == 0:
                continue
            if i % 25 == 0:
               print(f"Event {i}...")
            # Convert to pandas DataFrame for efficient group operations
            features = pd.DataFrame(graph['nodes'])
            features['key'] = graph['keys']
            features['charge'] = graph['charges']
            features['decay_id'] = graph['PrimaryHeavyHadronIndex']

            senders = graph['senders']
            receivers = graph['receivers']
            edge_feats = pd.DataFrame(graph['edges'], columns=['theta', 'trdist', 'DOCA', 'delta_z0'])
            edge_feats['sender_key'] = [graph['keys'][s] for s in senders]
            edge_feats['receiver_key'] = [graph['keys'][r] for r in receivers]

            # Separate chargedtree and neutral
            chargedtree_df = features[(features['charge'] != 0) & (features['decay_id'] >= 0)].copy()  # Ajoute la condition `decay_id >= 0`
            neutral_df = features[features['charge'] == 0].copy()

            # --- CHARGEDTREE NODE FEATURES ---
            group = chargedtree_df.groupby('decay_id')
            chargedtree_nodes = group.agg({
                5: 'sum',   # pz
                6: 'sum',   # pt
                7: 'mean'   # eta
            }).rename(columns={5: 'sum_pz', 6: 'sum_pt', 7: 'mean_eta'})

            # Mean DOCA, theta, trdist per decay tree
            edge_feats['pair'] = list(zip(edge_feats['sender_key'], edge_feats['receiver_key']))
            edge_feats_flip = edge_feats.copy()
            edge_feats_flip['pair'] = list(zip(edge_feats_flip['receiver_key'], edge_feats_flip['sender_key']))
            all_edges = pd.concat([edge_feats, edge_feats_flip])
            decay_pairs = chargedtree_df[['key', 'decay_id']].rename(columns={'key': 'chargedtree_key'})
            all_edges = all_edges.merge(decay_pairs, left_on='sender_key', right_on='chargedtree_key', how='inner')
            all_edges = all_edges.merge(decay_pairs, left_on='receiver_key', right_on='chargedtree_key', how='inner', suffixes=('_s', '_r'))
            same_decay = all_edges[all_edges['decay_id_s'] == all_edges['decay_id_r']]
            intra_means = same_decay.groupby('decay_id_s')[['DOCA', 'theta', 'trdist']].mean()
            chargedtree_nodes = chargedtree_nodes.join(intra_means, how='left').fillna(0).reset_index()

            chargedtree_node_feats = torch.tensor(
                chargedtree_nodes[['sum_pt', 'sum_pz', 'mean_eta', 'DOCA', 'theta', 'trdist']].values,
                dtype=torch.float
            )

            # --- NEUTRAL NODE FEATURES ---
            neutral_node_feats = torch.tensor(neutral_df[[6, 5, 7]].values, dtype=torch.float)  # pt, pz, eta

            # --- EDGE INDEX + ATTRIBUTES (vectorized) ---
            neutral_decay_pairs = pd.merge(
                neutral_df[['key']].assign(tmp=1),
                chargedtree_nodes[['decay_id']].assign(tmp=1),
                on='tmp'
            ).drop(columns='tmp').rename(columns={'key': 'neutral_key'})

            chargedtree_keys_per_decay = chargedtree_df[['key', 'decay_id']].rename(columns={'key': 'chargedtree_key'})

            extended = pd.merge(neutral_decay_pairs, chargedtree_keys_per_decay, on='decay_id')
            extended['pair_key'] = list(zip(extended['neutral_key'], extended['chargedtree_key']))

            edge_feats['pair_key'] = list(zip(edge_feats['sender_key'], edge_feats['receiver_key']))
            edge_feats_flip = edge_feats.copy()
            edge_feats_flip['pair_key'] = list(zip(edge_feats_flip['receiver_key'], edge_feats_flip['sender_key']))
            all_edge_feats = pd.concat([edge_feats, edge_feats_flip])

            joined = pd.merge(extended, all_edge_feats, on='pair_key')
            agg = joined.groupby(['neutral_key', 'decay_id'])[['DOCA', 'theta', 'trdist']].mean().reset_index()

            decay_id_to_idx = {d: i for i, d in enumerate(chargedtree_nodes['decay_id'])}
            key_to_idx_neutral = {k: i for i, k in enumerate(neutral_df['key'])}

            edge_index = []
            edge_attr = []
            edge_neutral_key = []
            edge_chargedtree_decay_id = []


            for _, row in agg.iterrows():
                n_idx = key_to_idx_neutral[row['neutral_key']]
                c_idx = decay_id_to_idx[row['decay_id']]
                c_feat = chargedtree_node_feats[c_idx]
                n_feat = neutral_node_feats[n_idx]
                attr = [
                    c_feat[0].item(),  # sum_pt
                    c_feat[1].item(),  # sum_pz
                    abs(c_feat[0].item() - n_feat[0].item()),
                    abs(c_feat[1].item() - n_feat[1].item()),
                    row['DOCA'],
                    row['theta'],
                    row['trdist']
                ]
                edge_index.append([c_idx, n_idx])
                edge_attr.append(attr)
                edge_neutral_key.append(row['neutral_key'])  # string ou int selon ton format
                edge_chargedtree_decay_id.append(row['decay_id'])


            if not edge_index:
                continue

            edge_index = torch.tensor(edge_index).T
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)

            # --- GLOBAL FEATURES ---
            globals_ = torch.tensor([
                len(neutral_node_feats),
                features.shape[0],
                chargedtree_nodes.shape[0],
                neutral_node_feats[:, 0].sum(),
                neutral_node_feats[:, 1].sum()
            ], dtype=torch.float).unsqueeze(0)

            data = HeteroData()
            data['chargedtree'].x = chargedtree_node_feats
            data['chargedtree'].decay_id = torch.tensor(chargedtree_nodes['decay_id'].values, dtype=torch.long)
            data['neutrals'].x = neutral_node_feats
            data['neutrals'].decay_id = torch.tensor(neutral_df['decay_id'].values, dtype=torch.long)
            data['chargedtree', 'to', 'neutrals'].edge_index = edge_index
            data['chargedtree', 'to', 'neutrals'].edges = edge_attr
            data['chargedtree', 'to', 'neutrals'].edge_chargedtree_decay_id = torch.tensor(edge_chargedtree_decay_id, dtype=torch.long)
            data['chargedtree', 'to', 'neutrals'].edge_neutral_key = torch.tensor(edge_neutral_key, dtype=torch.long)
            data['globals'].x = globals_

            # --- LABELS basés sur FromSameHeavyHadron dans graph_target ---
            graph_target = np.load(self.filenames_target[i], allow_pickle=True).item()

            # Récupère la mapping key → original index
            key_to_original_idx = dict(zip(graph['keys'], range(len(graph['keys']))))

            # Table de correspondance (i, j) -> FromSameHeavyHadron
            truth_pair_to_label = {}
            for s, r, edge_attr in zip(graph_target['senders'], graph_target['receivers'], graph_target['edges']):
                label = edge_attr[0]  # FromSameHeavyHadron
                truth_pair_to_label[(s, r)] = label
                truth_pair_to_label[(r, s)] = label  # symétrie

            matched_labels = []

            for c_idx, n_idx in edge_index.T.tolist():
                n_key = neutral_df.iloc[n_idx]['key']
                decay_id = chargedtree_nodes.iloc[c_idx]['decay_id']
                c_keys = chargedtree_df[chargedtree_df['decay_id'] == decay_id]['key'].tolist()

                label = 0
                n_orig_idx = key_to_original_idx.get(n_key, -1)

                for c_key in c_keys:
                    c_orig_idx = key_to_original_idx.get(c_key, -1)
                    if (n_orig_idx, c_orig_idx) in truth_pair_to_label:
                        label = truth_pair_to_label[(n_orig_idx, c_orig_idx)]
                        break  # une correspondance suffit

                matched_labels.append(label)

            data['chargedtree', 'to', 'neutrals'].y = torch.tensor(matched_labels, dtype=torch.float).unsqueeze(-1)
            


            # if self.performance_mode:
            #     data['init_senders'] = torch.from_numpy(graph["init_y"]["senders"]).long()
            #     data['init_receivers'] = torch.from_numpy(graph["init_y"]["receivers"]).long()
            #     data['init_y'] = torch.from_numpy(graph["init_y"]["edges"])
            #     data['init_keys'] = torch.from_numpy(graph["init_keys"])
            #     data['init_moth_ids'] = torch.from_numpy(graph["init_ids"])
            #     data['init_partids'] = torch.from_numpy(graph["init_part_ids"])
            #     data['final_keys'] = torch.from_numpy(graph["keys"])
            #     data['moth_ids'] = torch.from_numpy(graph["ids"])
            #     data['part_ids'] = torch.from_numpy(graph["part_ids"])
            #     data['lca_chain'] = torch.from_numpy(graph["lca_chain"])
            #     data['truth_senders'] = torch.from_numpy(graph["truth_senders"]).long()
            #     data['truth_receivers'] = torch.from_numpy(graph["truth_receivers"]).long()
            #     data['truth_y'] = torch.from_numpy(graph["truth_y"])
            #     data['truth_moth_ids'] = torch.from_numpy(graph["truth_ids"])
            #     data['truth_part_ids'] = torch.from_numpy(graph["truth_part_ids"])
            #     data['truth_part_keys'] = torch.from_numpy(graph["truth_part_keys"])
            #     init_senders = data.init_keys[data.init_senders]
            #     init_receivers = data.init_keys[data.init_receivers]
            #     senders =  data.final_keys[torch.from_numpy(old_senders).long()]
            #     receivers =  data.final_keys[torch.from_numpy(old_receivers).long()]
            #     init_cantor = 0.5 * (init_senders + init_receivers - 2) * (
            #                 init_senders + init_receivers - 1) + init_senders
            #     final_cantor = 0.5 * (senders + receivers - 2) * (senders + receivers - 1) + senders
            #     data['old_y'] = data.init_y[~torch.isin(init_cantor, final_cantor)]

            data_set.append(data)

        return data_set