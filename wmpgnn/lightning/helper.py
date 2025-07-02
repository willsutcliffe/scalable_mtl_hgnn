# mainly because I want to use pytorch lightning and dont want to import the class
import pandas as pd
import numpy as np

import torch

from particle import Particle

from wmpgnn.performance.reconstruction import reconstruct_decay, flatten, match_decays


def make_loggable(hparams_dict):
    loggable = {}
    for k, v in hparams_dict.items():
        if isinstance(v, torch.Tensor):
            if v.ndim == 0:
                loggable[k] = v.item()  # convert scalar tensor to float
            else:
                loggable[k] = v.tolist()  # convert vector/matrix to list
        else:
            loggable[k] = str(v)  # fallback to string
    return loggable


def get_ref_signal(ref_signal):  # Here we can define them all
    if 'Bs_JpsiPhi':
        signal_decay = {'daughters': ['mu+', 'mu-', 'K+', 'K-'], 'mothers': ['B(s)0']}
        cc_signal_decay = {'daughters': ['mu+', 'mu-', 'K+', 'K-'], 'mothers': ['B(s)~0']}
        return (signal_decay, cc_signal_decay)
    return {}


def particle_name(id_):
    if id_ == 0:
        return 'ghost'
    elif id_ == 10413:
        return 'D1(2420)+'
    elif id_ == -10413:
        return 'D1(2420)-'
    elif id_ == 4422:
        return 'Chi_cc++'
    elif id_ == -4422:
        return 'Chi_cc--'
    elif id_ == 4432:
        return 'Omega_cc++'
    elif id_ == -4432:
        return 'Omega_cc--'
    else:
        return Particle.from_pdgid(id_).name


def lca_reco_matrix(graph):
    senders = graph[('tracks', 'to', 'tracks')].edge_index[0].cpu()
    receivers = graph[('tracks', 'to', 'tracks')].edge_index[1].cpu()
    edges = graph[('tracks', 'to', 'tracks')].edges.cpu().numpy()

    edge_index = torch.vstack([senders, receivers])
    pd_matrix = pd.DataFrame(np.vstack(
        (edge_index[0], edge_index[1])).transpose(), columns=['senders', 'receivers'])
    pd_matrix["LCA_probs"] = list(edges)
    pd_matrix["LCA_dec"] = list(np.argmax(edges, axis=-1))  # LCA decision
    pd_matrix.set_index(['senders', 'receivers'], inplace=True)
    pd_matrix = pd_matrix.reset_index()
    pd_matrix = pd_matrix[pd_matrix['senders'] < pd_matrix['receivers']]
    reverse_order_indices = list(map(tuple, np.vstack((graph["receivers"], graph["senders"])).transpose()))
    return pd_matrix


def lca_truth_matrix(graph):
    senders = graph.truth_senders.cpu()
    receivers = graph.truth_receivers.cpu()
    init_y = graph["truth_y"].cpu()

    truth_lca = pd.DataFrame(np.column_stack((senders, receivers)), columns=['senders', 'receivers'])
    truth_lca['LCA_dec'] = np.reshape(
        np.argmax(
            np.reshape(init_y, (init_y.shape[0], 4)), axis=-1),
        (-1,))
    truth_lca = truth_lca[truth_lca['senders'] < truth_lca['receivers']]
    truth_lca['LCA_id_label'] = list(map(particle_name, graph['truth_moth_ids'].cpu().numpy()))
    truth_lca['LCA_id'] = graph['truth_moth_ids'].cpu().numpy()
    truth_lca['TrueFullChainLCA'] = graph['lca_chain'].cpu()
    return truth_lca


def get_pred_ft(graph, cluster, ft_score):  # node weights
    ft_score = ft_score.cpu()

    cluster_keys = cluster['node_keys']
    keys = graph['final_keys']

    b_daugthers_mask = np.isin(keys, cluster_keys)
    ft_bbar_score, ft_no_score, ft_b_score = ft_score[b_daugthers_mask].mean(dim=0)
    # here we want to pass the node weights [0, 1] and weight them for FT decision
    return ft_bbar_score, ft_no_score, ft_b_score


def get_b_cand_pv(graph, cluster, true_cluster, pv_score):
    pv_score = pv_score.squeeze().cpu()

    reco_cluster_keys = cluster['node_keys']
    true_cluster_keys = true_cluster['node_keys']
    keys = graph['final_keys']

    # Get the b daugthers idx and number
    b_daugthers_reco_idx = np.concatenate(np.argwhere(np.isin(keys, reco_cluster_keys) == True))
    n_reco_daugthers = b_daugthers_reco_idx.size
    b_daugthers_true_idx = np.concatenate(np.argwhere(np.isin(keys, true_cluster_keys) == True))
    n_true_daugthers = b_daugthers_true_idx.size

    # obtain their pv score
    edges = graph[("tracks", "to", "pvs")]["edge_index"][0].numpy()
    n_pvs = np.unique(graph[("tracks", "to", "pvs")]["edge_index"][1].numpy()).size
    reco_pv_edges = np.isin(edges, b_daugthers_reco_idx)
    true_pv_edges = np.isin(edges, b_daugthers_true_idx)

    reco_score = pv_score[reco_pv_edges].reshape(n_reco_daugthers, n_pvs)
    true_score = pv_score[true_pv_edges].reshape(n_true_daugthers, n_pvs)

    reco_pv_idx = torch.argmax(torch.sum(reco_score, dim=0)).item()
    true_pv_idx = torch.argmax(torch.sum(true_score, dim=0)).item()

    return reco_pv_idx, true_pv_idx


def eval_reco_performance(output, graph, event, signal_df, event_df, ft_score, pv_score, ref_signal):
    ref_signal = get_ref_signal(ref_signal)

    # do the eval on cpu
    output = output.cpu()  # is the model output
    graph = graph.cpu()  # is the inital graph, need to take the truth infromation

    # Check if B exists in the reco event
    Bparts = float(torch.sum(torch.argmax(graph['tracks', 'to', 'tracks'].y, -1) > 0))
    if Bparts < 1:
        print("no B reco")
        # continue  # but should still continue, check if also the reco found no B

    # Reco
    reco_LCA = lca_reco_matrix(output)
    particle_keys = list(output["final_keys"].numpy())
    total_number_of_particles = len(particle_keys)
    reco_cluster_dict, reco_num_clusters_per_order, _ = reconstruct_decay(reco_LCA, particle_keys)
    # check some stuff here
    # True cluster dict
    true_LCA = lca_truth_matrix(graph)
    particle_keys = list(graph["truth_part_keys"].numpy())
    particle_ids = list(map(particle_name, graph['truth_part_ids'].numpy()))
    truth_cluster_dict, truth_num_clusters_per_order, max_full_chain_depth_in_event = reconstruct_decay(true_LCA,
                                                                                                        particle_keys,
                                                                                                        particle_ids=particle_ids,
                                                                                                        truth_level_simulation=1)

    if truth_cluster_dict != {}:
        # Get the keys from the final state particle of the heavy hadron decay 
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

        perfect_event_reconstruction = 1  # Flag to track if the full event is perfectly reco. also takes into account correct LCA
        if number_of_selected_background_particles > 0:
            # More particle selected then background particle -> Background particle selected
            perfect_event_reconstruction = 0

        for tc_firstkey in truth_cluster_dict.keys():
            signal_match = 1

            if ref_signal != None:
                labels = truth_cluster_dict[tc_firstkey]['labels']
                mothers = [label[3:] for label in labels if 'c' == label[0]]
                node_keys = truth_cluster_dict[tc_firstkey]['node_keys']
                daughters = [label.split(':')[1] for label in labels if int(label.split(':')[0][1:]) in node_keys]

                if match_decays(daughters, ref_signal[0]['daughters']) or match_decays(daughters,
                                                                                       ref_signal[1]['daughters']):
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

            # Check if the reco is matching the true construction
            number_of_signal_particles = len(truth_cluster_dict[tc_firstkey]['node_keys'])
            perfect_signal_reconstruction = 1
            if reco_cluster_dict == {}:
                perfect_signal_reconstruction = 0
            else:
                if tc_firstkey not in reco_cluster_dict.keys():
                    perfect_signal_reconstruction = 0
                else:
                    if ['node_keys'] != truth_cluster_dict[tc_firstkey][
                        'node_keys'] or reco_cluster_dict[tc_firstkey]['LCA_values'] != \
                            truth_cluster_dict[tc_firstkey]['LCA_values']:
                        perfect_signal_reconstruction = 0
            perfect_event_reconstruction *= perfect_signal_reconstruction

            # Classify the reconstruction type of the event
            true_cluster = truth_cluster_dict[tc_firstkey]
            ft_bbar_score = -1
            ft_no_score = -1
            ft_b_score = -1
            perfect_reco = 0
            all_particles = 0
            none_iso = 0
            part_reco = 0
            none_associated = 0
            none_iso_n_bkg = 0
            reco_pv_idx = -1
            true_pv_idx = -1

            for cluster in reco_cluster_dict.values():
                true_in_reco = np.sum(np.isin(true_cluster['node_keys'], cluster['node_keys'])) / len(
                    true_cluster['node_keys'])
                if cluster['node_keys'] == true_cluster['node_keys']:
                    all_particles = 1
                    # Check if the LCA is matching as well
                    if cluster['LCA_values'] == true_cluster['LCA_values']:
                        perfect_reco = 1

                    # Get the flavour
                    ft_bbar_score, ft_no_score, ft_b_score = get_pred_ft(output, cluster, ft_score)
                    # Add PV reco and true PV
                    reco_pv_idx, true_pv_idx = get_b_cand_pv(output, cluster, true_cluster, pv_score)
                    break
                elif true_in_reco == 1 and len(cluster['node_keys']) > len(true_cluster['node_keys']):
                    none_iso = 1  # background tracks in signal

                    ft_bbar_score, ft_no_score, ft_b_score = get_pred_ft(output, cluster, ft_score)
                    none_iso_n_bkg = len(cluster['node_keys']) - len(
                        true_cluster['node_keys'])  # 'purity of bkg in non iso'
                    # Add PV reco and true PV
                    reco_pv_idx, true_pv_idx = get_b_cand_pv(output, cluster, true_cluster, pv_score)
                    break
                elif true_in_reco >= 0.2 and true_in_reco < 1:
                    part_reco = 1

                    ft_bbar_score, ft_no_score, ft_b_score = get_pred_ft(output, cluster, ft_score)
                    none_iso_n_bkg = len(cluster['node_keys']) - len(true_cluster['node_keys'])
                    # Add PV reco and true PV
                    reco_pv_idx, true_pv_idx = get_b_cand_pv(output, cluster, true_cluster, pv_score)
            if all_particles == 1:
                none_iso = 0
                part_reco = 0
            if none_iso == 1:
                part_reco = 0

            if all_particles == 0 and none_iso == 0 and part_reco == 0:
                none_associated = 1

            # Get the associated true ID
            indices = [particle_keys.index(x) for x in true_cluster['node_keys']]
            signal_LCA_id = true_LCA[true_LCA['senders'].isin(indices) | true_LCA['receivers'].isin(indices)]["LCA_id"]
            values, counts = np.unique(signal_LCA_id, return_counts=True)
            origin_B_id = values[np.argmax(counts)]
            # Log whether true B is reco or not
            # Easiest here is to asign the PV w/ corr and wrong
            signal_df = signal_df._append({'EventNumber': event,
                                           'NumParticlesInEvent': total_number_of_particles,
                                           'NumSignalParticles': number_of_signal_particles,
                                           'NumBkgParticles_noniso': none_iso_n_bkg,
                                           'PerfectSignalReconstruction': perfect_signal_reconstruction,
                                           'AllParticles': all_particles,
                                           'PerfectReco': perfect_reco,
                                           'NoneIso': none_iso,
                                           'PartReco': part_reco,
                                           'NotFound': none_associated,
                                           'SigMatch': signal_match,
                                           'Pred_FT_bbar_score': ft_bbar_score.item(),
                                           'Pred_FT_no_scrore': ft_no_score.item(),
                                           'Pred_FT_b_score': ft_b_score.item(),
                                           'B_id': origin_B_id,
                                           'reco_pv_idx': reco_pv_idx,
                                           'true_pv_idx': true_pv_idx
                                           }, ignore_index=True)

        # Add same PV frag particle found, add same PV opposite side B reco type, add signal B reco type
        # Log for event df
        if number_of_background_particles <= 0:
            number_of_background_particles = -1
        if number_of_particles_from_heavy_hadron <= 0:
            number_of_particles_from_heavy_hadron = -1
        event_df = event_df._append({'EventNumber': event,
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
                                     'NumTrueSignalsInEvent': len(truth_cluster_dict.keys()),
                                     'NumRecoSignalsInEvent': len(reco_cluster_dict.keys()),
                                     # 'TimeNodeFiltering': time_node_filtering,
                                     # 'TimeEdgeFiltering': time_edge_filtering,
                                     # 'TimeLCAReconstruction': time_LCA_reconstruction,
                                     # 'TimeSequence': time_node_filtering + time_edge_filtering + time_LCA_reconstruction,
                                     # 'TimeModel': time_model,
                                     # 'TimeReco': time_reco,
                                     # 'TimeTruth': time_truth
                                     },
                                    ignore_index=True)
    return signal_df, event_df


def intialize_test_df():
    signal_df = pd.DataFrame(
        columns=['EventNumber',
                 'NumParticlesInEvent', 'NumSignalParticles', 'NumBkgParticles_noniso',
                 'PerfectSignalReconstruction', 'AllParticles', 'PerfectReco', 'NoneIso', 'PartReco', 'NotFound',
                 'SigMatch',
                 'B_id', 'Pred_FT',
                 'reco_pv_idx', 'true_pv_idx'
                 ])
    """
    signal_df = signal_df.astype(
                {'EventNumber': np.int32, 'NumParticlesInEvent': np.int32, 'NumSignalParticles': np.int32, 'NumBkgParticles_noniso': np.int32,
                'PerfectSignalReconstruction': np.int32,
                'AllParticles': np.int32, 'PerfectReco': np.int32,
                'NoneIso': np.int32, 'PartReco': np.int32, 'NotFound': np.int32, 'SigMatch': np.int32, 'B_id': np.int32, 'Pred_FT': np.int32})
    """

    event_df = pd.DataFrame(
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
                 'TimeModel', 'TimeReco', 'TimeTruth'
                 ])

    return signal_df, event_df
