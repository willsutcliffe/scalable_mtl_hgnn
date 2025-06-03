import torch
import numpy as np
import pandas as pd
import time
import os
import math
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

def balance_edges(edge_index, edge_attr, edge_label, seed=42):
        torch.manual_seed(seed)
        # flatten labels to 1-D int tensor
        labels = edge_label.view(-1).long()
        sig_idx = (labels == 1).nonzero(as_tuple=True)[0]
        bkg_idx = (labels == 0).nonzero(as_tuple=True)[0]
        num_sig = sig_idx.numel()

        # if no signal or too peu de background, skip balancing
        if num_sig == 0 or bkg_idx.numel() < num_sig:
            return None  # on signalera qu'il faut skipper cet event

        # sample as many bkg as sig
        perm = torch.randperm(bkg_idx.numel())
        bkg_sample = bkg_idx[perm[:num_sig]]

        # concat and shuffle
        keep = torch.cat([sig_idx, bkg_sample], dim=0)
        keep = keep[torch.randperm(keep.numel())]

        # subset everything
        ei = edge_index[:, keep]
        ea = edge_attr[keep]
        el = edge_label[keep]

        return ei, ea, el

class CustomNeutralsHeteroDataset(Dataset):
    def __init__(self, filenames_input, filenames_target, performance_mode=False, config_loader=None, split="train"):
        self.filenames_input = filenames_input
        self.filenames_target = filenames_target
        self.performance_mode = performance_mode
        self.config_loader = config_loader
        self.split = split

    # No. of graphs
    def __len__(self):
        return len(self.filenames_target)

    def len(self):
        return len(self.filenames_target)

    def update(self, **kwargs):
        self.__dict__.update(kwargs)


    def get(self):
        start_time = time.time()
        dataset = []

        save_graph = self.config_loader.get("dataset.save_graph", False)
        load_graph = self.config_loader.get("dataset.load_graph", False)
        dir = self.config_loader.get("dataset.data_dir")
        balanced = self.config_loader.get("dataset.balanced_classes", False)
        subdir = "graphs_balanced" if balanced else "graphs"
        cache_dir = os.path.join(dir, subdir, f"{self.split}_graphs/")
        os.makedirs(cache_dir, exist_ok=True)
        # cache_path = os.path.join(cache_dir, f"{self.split}_graphs.pt")
        evt_max = self.config_loader.get(f"dataset.evt_max_{self.split}", None)
        chunk_size = 100
        total_events = len(self.filenames_input)
        if evt_max is not None:
            total_events = min(total_events, evt_max)
        num_chunks_needed = math.ceil(total_events / chunk_size)
        def get_cache_file(start_idx, end_idx):
            return os.path.join(cache_dir, f"events_{start_idx:05d}_to_{end_idx:05d}_{self.split}.pt")


        ### We can load the graphs if already generated
        if load_graph:
            print(f"Loading preprocessed graphs for split {self.split}")
            if balanced :
                print("Random background neutral particles have been discarded to have balanced class")
            for i in range(num_chunks_needed):
                cache_file = get_cache_file(i*chunk_size, min(((i + 1)*chunk_size), total_events) - 1)
                if not os.path.exists(cache_file):
                    raise RuntimeError(f"Missing cache chunk {cache_file}. Cannot load full dataset.")
                dataset.extend(torch.load(cache_file, weights_only=False))
            dataset = dataset[:total_events]
            total = time.time() - start_time
            print(f"Loaded {len(dataset)} graphs from cache in {num_chunks_needed} files (in {total:.2f}s).")
            return dataset

        ## Or generate the graphs
        print("Generating graphs from scratch...")
        if balanced :
            print("Discarding random background neutral particles to have balanced class")
        col_names = ['xProd','yProd','zProd','px','py','pz','pt','eta','charge', 'ParticleRecoType']
        for i in range(0, total_events, chunk_size):
            chunk_data = []
            for j in range(i, min(i + chunk_size, total_events)):
                in_fn = self.filenames_input[j]
                tgt_fn = self.filenames_target[j]            
                event_start = time.time()
                graph = np.load(in_fn, allow_pickle=True).item()
                if graph['nodes'].shape[0] == 0:
                    continue
                if j % 25 == 0:
                    print(f"Event {j}...")

                # Node features
                features = pd.DataFrame(graph['nodes'], columns=col_names)
                features['key'] = graph['keys']
                features['charge'] = graph['charges']
                features['decay_id'] = graph['PrimaryHeavyHadronIndex']

                # Edge features + labels
                edge_feats = pd.DataFrame(graph['edges'], columns=['theta','trdist','DOCA','delta_z0'])
                senders = np.array(graph['senders']); receivers = np.array(graph['receivers'])
                keys_arr = np.array(graph['keys'])
                edge_feats['sender_key'] = keys_arr[senders]
                edge_feats['receiver_key'] = keys_arr[receivers]
                # load labels array in same order
                tgt = np.load(tgt_fn, allow_pickle=True).item()
                labels = np.array([e[0] for e in tgt['edges']])
                edge_feats['label'] = labels

                # Separate charged & neutral nodes
                charged_df = features[(features['charge'] != 0) & (features['decay_id'] >= 0)].copy()
                neutral_df = features[features['charge'] == 0].copy()

                # Charged-tree node aggregates
                charged_stats = charged_df.groupby('decay_id').agg(
                    sum_pz=('pz','sum'), sum_pt=('pt','sum'), mean_eta=('eta','mean')
                )
                # intra-decay-edge means
                ef = edge_feats.copy()
                ef_flip = ef.rename(columns={'sender_key':'receiver_key','receiver_key':'sender_key'})
                all_e = pd.concat([ef, ef_flip], ignore_index=True)
                ck = charged_df[['key','decay_id']]
                m1 = all_e.merge(ck.rename(columns={'key':'sent_key'}), left_on='sender_key', right_on='sent_key')
                m2 = m1.merge(ck.rename(columns={'key':'rec_key'}), left_on='receiver_key', right_on='rec_key', suffixes=('_s','_r'))
                same = m2[m2['decay_id_s']==m2['decay_id_r']]
                intra = same.groupby('decay_id_s')[['DOCA','theta','trdist']].mean()
                charged_nodes = charged_stats.join(intra, how='left').fillna(0).reset_index()
                charged_feats = torch.tensor(
                    charged_nodes[['sum_pt','sum_pz','mean_eta','DOCA','theta','trdist']].values,
                    dtype=torch.float
                )
                #px, py and sum(px,py,pz), theta wrt p_B

                # Neutral features
                neutral_feats = torch.tensor(neutral_df[['pt','pz','eta']].values, dtype=torch.float)

                #px, py

                # Build full cross between neutrals and charged trees
                neutral_keys = neutral_df[['key']].rename(columns={'key':'neutral_key'}).assign(tmp=1)
                charged_keys = charged_nodes[['decay_id']].rename(columns={'decay_id':'decay_id'}).assign(tmp=1)
                cross = pd.merge(neutral_keys, charged_keys, on='tmp').drop(columns='tmp')
                cross = cross.rename(columns={'decay_id':'decay_id'})
                charged_keys_map = charged_df[['key','decay_id']].rename(columns={'key':'charged_key'})
                cross = cross.merge(charged_keys_map, on='decay_id')

                # Prepare pair_key and merge edge_feats
                ef['pair'] = list(zip(ef['sender_key'],ef['receiver_key']))
                ef_flip['pair'] = list(zip(ef_flip['sender_key'],ef_flip['receiver_key']))
                all_ef = pd.concat([ef,ef_flip], ignore_index=True)
                cross['pair'] = list(zip(cross['neutral_key'],cross['charged_key']))
                joined = pd.merge(cross, all_ef, on='pair', how='left')

                # Aggregate per neutral-charged pair
                # agg = joined.groupby(['neutral_key','decay_id'])[['DOCA','theta','trdist','label']]
                agg = joined.groupby(['neutral_key','decay_id'])[['theta','label']]
                agg = agg.mean().reset_index()

                # Build edge index & attributes
                dec2idx = {d:i for i,d in enumerate(charged_nodes['decay_id'])}
                neu2idx = {k:i for i,k in enumerate(neutral_df['key'])}
                agg['c_idx'] = agg['decay_id'].map(dec2idx)
                agg['n_idx'] = agg['neutral_key'].map(neu2idx)
                edge_index = torch.tensor(agg[['c_idx','n_idx']].values.T, dtype=torch.long)
                cvals = charged_feats[agg['c_idx'].values]
                nvals = neutral_feats[agg['n_idx'].values]
                edge_attr = torch.stack([
                    cvals[:,0],cvals[:,1],
                    torch.abs(cvals[:,0]-nvals[:,0]),
                    torch.abs(cvals[:,1]-nvals[:,1]),
                    # torch.tensor(agg['DOCA'].values),
                    torch.tensor(agg['theta'].values),
                    # torch.tensor(agg['trdist'].values)
                ], dim=1)
                edge_labels = torch.tensor(agg['label'].values, dtype=torch.float).unsqueeze(-1)

                if edge_index.size(1)==0:
                    continue

                # Global features
                globals_ = torch.tensor([[neutral_feats.size(0), features.shape[0], charged_nodes.shape[0],
                                        neutral_feats[:,0].sum(), neutral_feats[:,1].sum()]], dtype=torch.float)

                if balanced:
                    result = balance_edges(edge_index, edge_attr, edge_labels.squeeze(-1))
                    if result is None:
                        # skip this event if on n'a pas assez de neutrals signal ou bkg
                        continue
                    edge_index, edge_attr, edge_labels = result
                    # restore shape of labels to [...,1]
                    edge_labels = edge_labels.unsqueeze(-1)

                data = HeteroData()
                data['chargedtree'].x = charged_feats
                data['chargedtree'].decay_id = torch.tensor(charged_nodes['decay_id'].values, dtype=torch.long)
                data['neutrals'].x = neutral_feats
                data['neutrals'].decay_id = torch.tensor(neutral_df['decay_id'].values, dtype=torch.long)
                data['chargedtree','to','neutrals'].edge_index = edge_index
                data['chargedtree','to','neutrals'].edges = edge_attr
                data['chargedtree','to','neutrals'].y = edge_labels
                data['chargedtree', 'to', 'neutrals'].edge_chargedtree_decay_id = torch.tensor(agg['decay_id'].values, dtype=torch.long)
                data['chargedtree', 'to', 'neutrals'].edge_neutral_key = torch.tensor(agg['neutral_key'].values, dtype=torch.long)
                data['globals'].x = globals_

                chunk_data.append(data)

            dataset.extend(chunk_data)
                # print(f"Processed Event {i} in {time.time() - event_start:.2f}s")
            if save_graph:
                cache_file = get_cache_file(i, min((i + chunk_size), total_events) - 1)
                print(f"Saving chunk {i // chunk_size} to {cache_file} with {len(chunk_data)} events")
                torch.save(chunk_data, cache_file)

        total = time.time() - start_time
        print(f"Processed {len(dataset)} events in {total:.2f}s, avg {total/max(len(dataset),1):.2f}s/event")
        return dataset


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