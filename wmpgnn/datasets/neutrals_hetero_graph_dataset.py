import torch
import numpy as np
import pandas as pd
import time
import os
import math
import itertools
from torch_geometric.data import Dataset, Data
from torch_geometric.data import HeteroData

def angle_phi(v1, v2):
    """
    Compute the angle (in radians) between two vectors v1 and v2 along dim=1.

    Parameters:
        v1 (Tensor): A tensor of shape (N, D) representing N vectors.
        v2 (Tensor): A tensor of shape (N, D) representing N vectors.

    Returns:
        Tensor: A tensor of shape (N,) containing the angles between v1 and v2.
    """
    dot = (v1 * v2).sum(dim=1)  # Dot product between vectors
    norm1 = v1.norm(dim=1)      # Norm of v1
    norm2 = v2.norm(dim=1)      # Norm of v2

    # Compute cosine of the angle and clamp to avoid numerical instability
    cos_theta = torch.clamp(dot / (norm1 * norm2 + 1e-8), -1.0, 1.0)

    return torch.acos(cos_theta)  # Return the angle in radians


def find_row_indices(t1, t2):
    """
    For each row in t1, find the index of the matching row in t2.
    Returns both the index and one-hot encoding of the matches.

    Parameters:
        t1 (Tensor): A tensor of shape (N1, D).
        t2 (Tensor): A tensor of shape (N2, D).

    Returns:
        indices (Tensor): A tensor of shape (N1,) where each entry is the index
                          of the matching row in t2, or -1 if no match found.
        one_hot_encoded (Tensor): A one-hot encoded tensor of shape (N1, N2).
                                  Rows with no match will have all zeros.
    """
    # Expand both tensors to compare every row pairwise
    t1_expanded = t1.unsqueeze(1)  # Shape: (N1, 1, D)
    t2_expanded = t2.unsqueeze(0)  # Shape: (1, N2, D)

    # Check for element-wise equality across rows
    matches = (t1_expanded == t2_expanded).all(dim=2)  # Shape: (N1, N2)

    # Convert matches to integers for use with argmax
    matches_int = matches.int()

    # Find index of first match in t2 for each row in t1
    indices = torch.argmax(matches_int, dim=1)

    # Mark rows with no match by setting index to -1
    valid = matches.any(dim=1)
    indices[~valid] = -1

    # Compute one-hot encoding (rows with -1 will be all zeros)
    num_classes = t2.shape[0]
    one_hot_encoded = torch.nn.functional.one_hot(
        torch.clamp(indices, min=0), num_classes=num_classes
    )

    # Zero-out rows with no match
    one_hot_encoded[~valid] = 0

    return indices, one_hot_encoded


def balance_edges(edge_index, edge_attr, edge_label, seed=42):
    """
    Balance signal and background edges by downsampling the background.

    Parameters:
        edge_index (Tensor): Tensor of shape (2, E) representing edge indices.
        edge_attr (Tensor): Tensor of shape (E, F) with edge features.
        edge_label (Tensor): Tensor of shape (E,) with binary labels (0: bkg, 1: sig).
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple[Tensor, Tensor, Tensor] or None:
            - edge_index: Balanced edge indices.
            - edge_attr: Balanced edge features.
            - edge_label: Balanced labels.
            Returns None if balancing is not possible.
    """
    torch.manual_seed(seed)

    # Flatten labels to 1D and get indices for each class
    labels = edge_label.view(-1).long()
    sig_idx = (labels == 1).nonzero(as_tuple=True)[0]
    bkg_idx = (labels == 0).nonzero(as_tuple=True)[0]
    num_sig = sig_idx.numel()

    # Skip balancing if no signal or not enough background
    if num_sig == 0 or bkg_idx.numel() < num_sig:
        return None  # Event should be skipped

    # Randomly sample background to match number of signal edges
    perm = torch.randperm(bkg_idx.numel())
    bkg_sample = bkg_idx[perm[:num_sig]]

    # Concatenate and shuffle signal and background indices
    keep = torch.cat([sig_idx, bkg_sample], dim=0)
    keep = keep[torch.randperm(keep.numel())]

    # Subset edge_index, edge_attr, and edge_label
    ei = edge_index[:, keep]
    ea = edge_attr[keep]
    el = edge_label[keep]

    return ei, ea, el

class CustomNeutralsHeteroDataset(Dataset):
    """
    Custom dataset for building or loading heterogeneous graphs with charged and neutral particles.

    The dataset can either:
    - Load preprocessed graphs from disk (if caching is enabled)
    - Generate graphs from raw `.npy` input files and labels

    Attributes:
        filenames_input (List[str]): List of paths to input graph files.
        filenames_target (List[str]): List of paths to target label files.
        performance_mode (bool): If True, optimizations can be used.
        config_loader (Any): Configuration manager for loading parameters.
        split (str): Dataset split name ("train", "val", "test", etc.).
        sizes (List[int]): [number of magup, number of magdown events], used if polarity='magall'.
    """

    def __init__(self, filenames_input, filenames_target, performance_mode=False, config_loader=None, split="train", sizes=[0, 0]):
        self.filenames_input = filenames_input
        self.filenames_target = filenames_target
        self.performance_mode = performance_mode
        self.config_loader = config_loader
        self.split = split
        self.sizes = sizes

    def __len__(self):
        """Return the number of target files (i.e., number of events)."""
        return len(self.filenames_target)

    def len(self):
        """Alias for __len__()."""
        return len(self.filenames_target)

    def update(self, **kwargs):
        """Update attributes of the dataset object with given keyword arguments."""
        self.__dict__.update(kwargs)

    def get(self):
        """
        Either loads graphs from cache or builds them from scratch based on input files.
        
        Returns:
            List[HeteroData]: List of heterogeneous graphs (chargedtree + neutrals).
        """
        start_time = time.time()
        dataset = []

        # Load options from config
        save_graph = self.config_loader.get("dataset.save_graph", False)
        load_graph = self.config_loader.get("dataset.load_graph", False)
        balanced = self.config_loader.get("dataset.balanced_classes", False)
        polarity = self.config_loader.get("dataset.polarity", False)
        neutrals_edges = "neutrals_neutrals" in self.config_loader.get("model.edge_types")

        evt_max = self.config_loader.get(f"dataset.evt_max_{self.split}", None)
        chunk_size = 100
        total_events = len(self.filenames_input)
        #print(total_events)
        if evt_max is not None:
            total_events = min(total_events, evt_max)

        def get_cache_file(pol, start_idx, end_idx):
            """
            Get the cache file path based on polarity and index range.
            """
            data_type =self.config_loader.get("dataset.data_type")

            data_subfolder =f"{data_type}_with_id"
            if pol in ['magup', 'magdown']:
                dir = os.path.join(self.config_loader.get("dataset.data_dir"), pol, data_subfolder)
            else:
                dir = os.path.join(self.config_loader.get("dataset.data_dir"), data_subfolder)
            subdir = "graphs"
            if balanced:
                subdir += "_balanced"
            if neutrals_edges:
                subdir += "_nedges"
            cache_dir = os.path.join(dir, subdir, f"{self.split}_graphs/")
            os.makedirs(cache_dir, exist_ok=True)
            return os.path.join(cache_dir, f"events_{start_idx:05d}_to_{end_idx:05d}_{self.split}.pt")

        ### Try loading graphs from cache if enabled
        if load_graph:
            max_event_name = f"dataset.evt_max_{self.split}"
            total_events = self.config_loader.get(max_event_name)
            num_chunks_needed = math.ceil(total_events / chunk_size)
            print(f"Loading preprocessed graphs for split {self.split}")
            if balanced:
                print("Random background neutral particles have been discarded to have balanced class")

            if polarity == 'magall':
                # Load graphs separately for magup and magdown
                total_events_up, total_events_down = self.sizes
                num_chunks_needed_up = math.ceil(total_events_up / chunk_size)
                num_chunks_needed_down = math.ceil(total_events_down / chunk_size)

                for i in range(num_chunks_needed_up):
                    cache_file = get_cache_file('magup', i * chunk_size, min((i + 1) * chunk_size, total_events_up) - 1)
                    if not os.path.exists(cache_file):
                        raise RuntimeError(f"Missing cache chunk {cache_file}. Cannot load full dataset.")
                    # print(f'Path for magup: {cache_file}')
                    dataset.extend(torch.load(cache_file, weights_only=False))

                for i in range(num_chunks_needed_down):
                    cache_file = get_cache_file('magdown', i * chunk_size, min((i + 1) * chunk_size, total_events_down) - 1)
                    if not os.path.exists(cache_file):
                        raise RuntimeError(f"Missing cache chunk {cache_file}. Cannot load full dataset.")
                    # print(f'Path for magdown: {cache_file}')
                    dataset.extend(torch.load(cache_file, weights_only=False))

            elif polarity in ('magdown', 'magup', 'PYTHIA'):
                for i in range(num_chunks_needed):
                    cache_file = get_cache_file(polarity, i * chunk_size, min((i + 1) * chunk_size, total_events) - 1)
                    if not os.path.exists(cache_file):
                        raise RuntimeError(f"Missing cache chunk {cache_file}. Cannot load full dataset.")
                    # print(f'Path : {cache_file}')
                    dataset.extend(torch.load(cache_file, weights_only=False))
            
            # elif polarity == 'PYTHIA':
            #     for i in range(num_chunks_needed):
            #         cache_file = get_cache_file(polarity, i * chunk_size, min((i + 1) * chunk_size, total_events) - 1)
            #         if not os.path.exists(cache_file):
            #             raise RuntimeError(f"Missing cache chunk {cache_file}. Cannot load full dataset.")
            #         # print(f'Path : {cache_file}')
            #         dataset.extend(torch.load(cache_file, weights_only=False))
            else:
                raise Exception(f"Unexpected magnet polarity {polarity}. Please use magdown, magup or magall. You can also set PYTHIA for the simplified simulation.")

            dataset = dataset[:total_events]
            total = time.time() - start_time
            print(f"Loaded {len(dataset)} graphs from cache in {num_chunks_needed} files (in {total:.2f}s).")
            return dataset

        ### Otherwise, build graphs from scratch
        print("Generating graphs from scratch...")
        if balanced:
            print("Discarding random background neutral particles to have balanced class")


        col_names = ['xProd', 'yProd', 'zProd', 'px', 'py', 'pz', 'pt', 'eta', 'charge','ParticleType', 'ParticleRecoType', 'id']

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

                # === Load node and edge data ===
                features = pd.DataFrame(graph['nodes'], columns=col_names)
                features['key'] = graph['keys']
                features['charge'] = graph['charges']
                features['decay_id'] = graph['PrimaryHeavyHadronIndex']

                edge_feats = pd.DataFrame(graph['edges'], columns=['theta', 'trdist', 'DOCA', 'delta_z0'])
                senders = np.array(graph['senders'])
                receivers = np.array(graph['receivers'])
                keys_arr = np.array(graph['keys'])
                edge_feats['sender_key'] = keys_arr[senders]
                edge_feats['receiver_key'] = keys_arr[receivers]

                tgt = np.load(tgt_fn, allow_pickle=True).item()
                labels = np.array([e[0] for e in tgt['edges']])
                edge_feats['label'] = labels

                # === Split nodes ===
                charged_df = features[(features['charge'] != 0) & (features['decay_id'] >= 0)].copy()
                neutral_df = features[features['charge'] == 0].copy()

                # === Charged node aggregation ===
                charged_stats = charged_df.groupby('decay_id').agg(
                    sum_px=('px', 'sum'), sum_py=('py', 'sum'), sum_pz=('pz', 'sum'),
                    sum_pt=('pt', 'sum'), mean_eta=('eta', 'mean')
                )
                ef = edge_feats.copy()
                ef_flip = ef.rename(columns={'sender_key': 'receiver_key', 'receiver_key': 'sender_key'})
                all_e = pd.concat([ef, ef_flip], ignore_index=True)
                ck = charged_df[['key', 'decay_id']]
                m1 = all_e.merge(ck.rename(columns={'key': 'sent_key'}), left_on='sender_key', right_on='sent_key')
                m2 = m1.merge(ck.rename(columns={'key': 'rec_key'}), left_on='receiver_key', right_on='rec_key', suffixes=('_s', '_r'))
                same = m2[m2['decay_id_s'] == m2['decay_id_r']]
                intra = same.groupby('decay_id_s')[['DOCA', 'theta', 'trdist']].mean()
                charged_nodes = charged_stats.join(intra, how='left').fillna(0).reset_index()
                charged_feats = torch.tensor(
                    charged_nodes[['sum_px', 'sum_py', 'sum_pz', 'sum_pt', 'mean_eta', 'DOCA', 'theta', 'trdist']].values,
                    dtype=torch.float
                )

                # === Neutral features ===
                neutral_feats = torch.tensor(neutral_df[['px', 'py', 'pz', 'pt', 'eta']].values, dtype=torch.float)
                neutral_keys_nn = neutral_df['key'].values
                num_neutrals = len(neutral_keys_nn)
                neutral_id = torch.tensor(neutral_df[['id','ParticleRecoType']].values, dtype=torch.float)


                # === Add neutral-neutral edges ===
                if num_neutrals >= 2:
                    # Build DataFrame for all unique (i<j) neutral-neutral pairs
                    neutral_pairs = list(itertools.combinations(range(num_neutrals), 2))
                    idx1 = torch.tensor([i for i, j in neutral_pairs], dtype=torch.long)
                    idx2 = torch.tensor([j for i, j in neutral_pairs], dtype=torch.long)
                    key1 = neutral_keys_nn[idx1]
                    key2 = neutral_keys_nn[idx2]

                    pair_df = pd.DataFrame({
                        'sender_key': key1,
                        'receiver_key': key2,
                        'idx1': idx1.numpy(),
                        'idx2': idx2.numpy()
                    })
                    pair_df['pair'] = list(zip(pair_df['sender_key'], pair_df['receiver_key']))

                    # Create edge list from graph and flip
                    ef_nn = pd.DataFrame(graph['edges'], columns=['theta','trdist','DOCA','delta_z0'])
                    keys_arr_nn = np.array(graph['keys'])
                    ef_nn['sender_key'] = keys_arr_nn[np.array(graph['senders'])]
                    ef_nn['receiver_key'] = keys_arr_nn[np.array(graph['receivers'])]
                    ef_nn['pair'] = list(zip(ef_nn['sender_key'], ef_nn['receiver_key']))

                    ef_nn_flip = ef_nn.copy()
                    ef_nn_flip['pair'] = list(zip(ef_nn_flip['receiver_key'], ef_nn_flip['sender_key']))
                    all_ef_nn = pd.concat([ef_nn, ef_nn_flip], ignore_index=True)

                    # Merge and keep theta for existing pairs only
                    pair_df = pair_df.merge(all_ef_nn[['pair', 'theta']], on='pair', how='inner')

                    # Now we have filtered idx1/idx2 with aligned theta values
                    idx1 = torch.tensor(pair_df['idx1'].values, dtype=torch.long)
                    idx2 = torch.tensor(pair_df['idx2'].values, dtype=torch.long)
                    theta_vals = torch.tensor(pair_df['theta'].values, dtype=torch.float)

                    # Build edge_index
                    nn_edge_index = torch.stack([idx1, idx2], dim=0)

                    # Build features
                    p1 = neutral_feats[idx1]
                    p2 = neutral_feats[idx2]
                    nn_edge_attr = torch.stack([
                        p1[:, 0] + p2[:, 0],                         # sum_px
                        p1[:, 1] + p2[:, 1],                         # sum_py
                        p1[:, 2] + p2[:, 2],                         # sum_pz
                        p1[:, 3] + p2[:, 3],                         # sum_pt
                        torch.abs(p1[:, 0] - p2[:, 0]),              # |Δpx|
                        torch.abs(p1[:, 1] - p2[:, 1]),              # |Δpy|
                        torch.abs(p1[:, 2] - p2[:, 2]),              # |Δpz|
                        torch.abs(p1[:, 3] - p2[:, 3]),              # |Δpt|
                        theta_vals                                   # theta from edges
                    ], dim=1)
                else:
                    nn_edge_index = torch.empty((2, 0), dtype=torch.long)
                    nn_edge_attr = torch.empty((0, 9), dtype=torch.float)
                # === End neutral-neutral edge block ===


                # === Build cross features between charged trees and neutrals ===
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
                cval = charged_feats[agg['c_idx'].values]
                nval = neutral_feats[agg['n_idx'].values]
                edge_attr = torch.stack([
                    cval[:, 3] + nval[:, 3],                                # sum_pt
                    cval[:, 0] + nval[:, 0],                                # sum_px
                    cval[:, 1] + nval[:, 1],                                # sum_py
                    cval[:, 2] + nval[:, 2],                                # sum_pz
                    torch.abs(cval[:, 3] - nval[:, 3]),                     # |Δpt|
                    torch.abs(cval[:, 0] - nval[:, 0]),                     # |Δpx|
                    torch.abs(cval[:, 1] - nval[:, 1]),                     # |Δpy|
                    torch.abs(cval[:, 2] - nval[:, 2]),                     # |Δpz|
                    torch.tensor(agg['theta'].values),                      # theta mean (among charged part and neutral)
                    angle_phi(cval[:, 0:2], nval[:, 0:2])                   # phi (between heavy hadron and neutral)
                ], dim=1)
                edge_labels = torch.tensor(agg['label'].values, dtype=torch.float).unsqueeze(-1)
                neutrals_id_edges = torch.tensor(
                    neutral_df.iloc[agg['n_idx']][['id', 'ParticleRecoType']].values,
                    dtype=torch.float
                )

                # === Balance and assemble the final graph ===
                if edge_index.size(1) == 0:
                    continue

                globals_ = torch.tensor([[neutral_feats.size(0), features.shape[0], charged_nodes.shape[0],
                                        neutral_feats[:, 0].sum(), neutral_feats[:, 1].sum(),
                                        neutral_feats[:, 2].sum(), neutral_feats[:, 3].sum()]], dtype=torch.float)

                if balanced:
                    result = balance_edges(edge_index, edge_attr, edge_labels.squeeze(-1))
                    if result is None:
                        continue  # skip event if not enough signal/background
                    edge_index, edge_attr, edge_labels = result
                    edge_labels = edge_labels.unsqueeze(-1)

                data = HeteroData()
                data['chargedtree'].x = charged_feats
                data['chargedtree'].decay_id = torch.tensor(charged_nodes['decay_id'].values, dtype=torch.long)
                data['neutrals'].x = neutral_feats
                data['neutrals'].decay_id = torch.tensor(neutral_df['decay_id'].values, dtype=torch.long)
                data['neutrals'].id = neutral_id
                data['chargedtree', 'to', 'neutrals'].edge_index = edge_index
                data['chargedtree', 'to', 'neutrals'].edges = edge_attr
                data['chargedtree', 'to', 'neutrals'].y = edge_labels
                data['chargedtree', 'to', 'neutrals'].neutrals_id = neutrals_id_edges
                data['chargedtree', 'to', 'neutrals'].edge_chargedtree_decay_id = torch.tensor(agg['decay_id'].values, dtype=torch.long)
                data['chargedtree', 'to', 'neutrals'].edge_neutral_key = torch.tensor(agg['neutral_key'].values, dtype=torch.long)
                data['neutrals', 'to', 'neutrals'].edge_index = nn_edge_index
                data['neutrals', 'to', 'neutrals'].edges = nn_edge_attr
                data['globals'].x = globals_

                chunk_data.append(data)

            dataset.extend(chunk_data)

            if save_graph:
                cache_file = get_cache_file(polarity, i, min(i + chunk_size, total_events) - 1)
                print(f"Saving chunk {i // chunk_size} to {cache_file} with {len(chunk_data)} events")
                torch.save(chunk_data, cache_file)

        total = time.time() - start_time
        print(f"Processed {len(dataset)} events in {total:.2f}s, avg {total/max(len(dataset),1):.2f}s/event")
        return dataset

