import os
import glob
import pandas as pd
import numpy as np
from torch_geometric.data import Dataset
from wmpgnn.datasets.neutrals_hetero_graph_dataset import CustomNeutralsHeteroDataset

# === CONFIG ===
input_dir = "/eos/user/e/ebornand/DFEI/FullMC/npy_array/magdown/neutrals/training_dataset"
input_files = sorted(glob.glob(os.path.join(input_dir, "input_*.npy")))[:10]
target_files = sorted(glob.glob(os.path.join(input_dir, "target_*.npy")))[:10]
output_folder = "/afs/cern.ch/user/e/ebornand/DFEI_HGNN/csv_outputs"
os.makedirs(output_folder, exist_ok=True)

# === INSTANTIATE DATASET ===
dataset = CustomNeutralsHeteroDataset(input_files, target_files, performance_mode=False, n_classes=2)
data_list = dataset.get()
print(f"Data imported, start processing...")

for idx, data in enumerate(data_list):
    raw_input = np.load(input_files[idx], allow_pickle=True).item()
    raw_target = np.load(target_files[idx], allow_pickle=True).item()

    keys = np.array(raw_input['keys'])
    charges = np.array(raw_input['charges'])
    decay_ids = np.array(raw_input['PrimaryHeavyHadronIndex'])
    is_charged = charges != 0
    is_neutral = ~is_charged

    charged_keys = keys[is_charged]
    charged_decay_ids = decay_ids[is_charged]
    neutral_keys = keys[is_neutral]
    neutral_decay_ids = decay_ids[is_neutral]

    output_excel = os.path.join(output_folder, f"event_{idx}_summary.xlsx")
    with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:

        # === chargedtree nodes ===
        chargedtree_df = pd.DataFrame(data['chargedtree'].x.numpy(), columns=[
            'sum_pt', 'sum_pz', 'mean_eta', 'mean_DOCA', 'mean_theta', 'mean_trdist'])
        chargedtree_df.insert(0, 'decay_id', data['chargedtree'].decay_id.numpy())
        chargedtree_df.to_excel(writer, sheet_name='chargedtree_nodes', index_label='chargedtree_idx')

        # === Neutral nodes ===
        neutral_df = pd.DataFrame(data['neutrals'].x.numpy(), columns=['pt', 'pz', 'eta'])
        # neutral_df.insert(0, 'key', data['keys'].decay_id.numpy())
        neutral_df.insert(1, 'decay_id', data['neutrals'].decay_id.numpy())
        neutral_df.to_excel(writer, sheet_name='neutral_nodes', index_label='neutral_idx')

        # === Edges ===
        edge_index = data['chargedtree', 'to', 'neutrals'].edge_index.T.numpy()
        edges = data['chargedtree', 'to', 'neutrals'].edges.numpy()

        neutral_idx = edge_index[:, 0]
        chargedtree_idx = edge_index[:, 1]
        # neutral_keys_mapped = neutral_keys[neutral_idx]
        # chargedtree_keys_mapped = charged_keys[chargedtree_idx]

        edge_df = pd.DataFrame(edges, columns=[
            'sum_pt', 'sum_pz', 'abs_pt_diff', 'abs_pz_diff', 'mean_DOCA', 'mean_theta', 'mean_trdist'])

        edge_df.insert(0, 'edge_idx', range(len(edge_df)))
        edge_df.insert(1, 'neutral_idx', neutral_idx)
        edge_df.insert(2, 'chargedtree_idx', chargedtree_idx)
        edge_df.insert(3, 'chargedtree_decay_id', data['chargedtree', 'to', 'neutrals'].edge_chargedtree_decay_id.numpy())
        edge_df.insert(4, 'neutral_key', data['chargedtree', 'to', 'neutrals'].edge_neutral_key.numpy())

        edge_df.to_excel(writer, sheet_name='edges', index=False)

        # === Labels ===
        y = data['chargedtree', 'to', 'neutrals'].y.numpy().flatten()
        pd.DataFrame({'label': y}).to_excel(writer, sheet_name='edge_labels', index_label='edge_idx')

        # === Global features ===
        global_feats = pd.DataFrame(data['globals'].x.numpy(), columns=[
            'n_neutrals', 'n_particles', 'n_trees', 'sum_pt_neutrals', 'sum_pz_neutrals'])
        global_feats.to_excel(writer, sheet_name='global_features', index=False)

        # === Raw metadata (keys/charges/decay ids) ===
        pd.DataFrame(raw_input['nodes']).to_excel(writer, sheet_name='raw_nodes', index=False)
        pd.DataFrame({
            'keys': raw_input['keys'],
            'charges': raw_input['charges'],
            'PrimaryHeavyHadronIndex': raw_input['PrimaryHeavyHadronIndex']
        }).to_excel(writer, sheet_name='raw_metadata', index=False)

        # === Raw target connections ===
        pd.DataFrame({
            'sender': np.array(raw_target['senders']).flatten(),
            'receiver': np.array(raw_target['receivers']).flatten(),
            'label': np.array(raw_target['edges']).flatten()
        }).to_excel(writer, sheet_name='raw_target_edges', index=False)

    print(f"Saved event {idx} summary to {output_excel}")
