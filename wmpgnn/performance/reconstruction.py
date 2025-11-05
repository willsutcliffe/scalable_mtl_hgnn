import torch

import pandas as pd
import numpy as np

from wmpgnn.performance.reco_helper import reconstruct_decay, flatten, match_decays, particle_name


def get_ref_signal(ref_signal):  # Here we can define them all
    if ref_signal == 'Bs_Jpsiphi':
        signal_decay = {'daughters': ['mu+', 'mu-', 'K+', 'K-'], 'mothers': ['B(s)0']}
        cc_signal_decay = {'daughters': ['mu+', 'mu-', 'K+', 'K-'], 'mothers': ['B(s)~0']}
        return signal_decay, cc_signal_decay
    elif ref_signal == 'Bd_JpsiKs':  # Check if this is cor
        signal_decay = {'daughters': ['mu+', 'mu-', 'pi+', 'pi-'], 'mothers': ['B0']}
        cc_signal_decay = {'daughters': ['mu+', 'mu-', 'pi+', 'pi-'], 'mothers': ['B~0']}
        return signal_decay, cc_signal_decay
    elif ref_signal == "Bs_Dspi":
        signal_decay = {'daughters': ['K+', 'K-', 'pi+', 'pi-'], 'mothers': ['B(s)0']}
        cc_signal_decay = {'daughters': ['K+', 'K-', 'pi+', 'pi-'], 'mothers': ['B(s)~0']}
        return signal_decay, cc_signal_decay
    return {}


def lca_reco_matrix(graph, mode="reco"):
    edge_index = graph[('tracks', 'to', 'tracks')].edge_index.cpu()

    pd_matrix = pd.DataFrame(edge_index.T, columns=['senders', 'receivers'])
    if mode == "reco":
        edges = graph[('tracks', 'to', 'tracks')].lca.cpu()
        pd_matrix["LCA_dec"] = torch.argmax(edges, axis=-1).tolist()  # LCA decision
    else:
        pd_matrix["LCA_dec"] = graph[('tracks', 'to', 'tracks')].y.tolist()  # LCA decision
    pd_matrix.set_index(['senders', 'receivers'], inplace=True)
    pd_matrix = pd_matrix.reset_index()
    pd_matrix = pd_matrix[pd_matrix['senders'] < pd_matrix['receivers']]
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


def get_asso_frag(sig_dict, graph, cluster):
    res_dict = sig_dict

    cluster_keys = cluster['node_keys']
    keys = graph['final_keys']
    b_daugthers_mask = np.isin(keys, cluster_keys)
    b_idx = str(graph["tracks"].asso_hh[b_daugthers_mask][0].item())

    int_list = graph["frag_y"].tolist()
    first_two = np.array([str(num)[:2] for num in int_list])
    rest_digits = np.array([str(num)[2:] if len(str(num)) > 2 else '' for num in int_list])

    frag_selbool = rest_digits == b_idx
    res_dict["frags"] = "_".join(first_two[frag_selbool])
    res_dict["frags_pid"] = "_".join(str(x.item()) for x in graph["frag_pid"][rest_digits == b_idx])
    return res_dict


def get_pred_ft(sig_dict, graph, cluster, ft_score):
    res_dict = sig_dict
    # Save combined b bbar score, save individual scores, save pid of final
    cluster_keys = cluster['node_keys']
    keys = graph['final_keys']
    b_daugthers_mask = np.isin(keys, cluster_keys)

    # Get the pid of the particles
    res_dict["final_pid"] = ','.join(str(x.item()) for x in graph["part_ids"][b_daugthers_mask])

    # Get the individual scores stored as strings
    ft_score = ft_score[b_daugthers_mask].cpu()
    res_dict["final_b_score"] = ','.join(str(x.item()) for x in ft_score[:, :1].squeeze())
    res_dict["final_bbar_score"] = ','.join(str(x.item()) for x in ft_score[:, 2:].squeeze())

    res_dict["ft_b_score"], _, res_dict["ft_bbar_score"] = ft_score.mean(dim=0).tolist()

    return res_dict


def reco_event(graph, event, config, signal, sig_df, evt_df, ft_des):
    ref_signal = get_ref_signal(signal)
    graph = graph.cpu()

    """Check if reco B exist and true B exist"""
    if torch.sum(graph['tracks', 'to', 'tracks'].y) == 0:
        print("no true B exist")
    if config["LCA"] and torch.sum(torch.argmax(graph['tracks', 'to', 'tracks'].edges, dim=-1)) == 0:
        print("no reco B candidate found")

    """Obtain the LCA scores of both true and reco graphs"""
    if config["LCA"]:
        reco_LCA = lca_reco_matrix(graph, mode="reco")
    else:
        reco_LCA = lca_reco_matrix(graph, mode="true")
    true_LCA = lca_truth_matrix(graph)

    particle_keys = graph["final_keys"].tolist()
    n_part = len(particle_keys)
    rc_dict, r_nclust_order, _ = reconstruct_decay(reco_LCA, particle_keys)

    particle_keys = graph["truth_part_keys"].tolist()
    particle_ids = list(map(particle_name, graph['truth_part_ids'].numpy()))
    tc_dict, t_nclust_order, max_chain_depth = reconstruct_decay(true_LCA, particle_keys,
                                                                 particle_ids=particle_ids, truth_level_simulation=1)
    if tc_dict != {}:
        part_heavy_h = flatten([tc_dict[tc_firstkey]['node_keys'] for tc_firstkey in tc_dict.keys()])
        n_part_heavy_h = len(part_heavy_h)
        n_bkg_part = n_part - n_part_heavy_h

        if rc_dict != {}:
            sel_part = flatten(
                [rc_dict[tc_firstkey]['node_keys'] for tc_firstkey in rc_dict.keys()])
            n_sel_part = len(sel_part)
            n_sel_heavy_h = len(list(set(sel_part).intersection(part_heavy_h)))
            n_sel_bkg_part = n_sel_part - n_sel_heavy_h
        else:
            n_sel_part = 0
            n_sel_heavy_h = 0
            n_sel_bkg_part = 0

        perfect_evt_reco = 1  # Flag for perfect event reco
        if n_sel_bkg_part > 0:
            perfect_evt_reco = 0

        # Looping over reco candidates
        for tc_key in tc_dict.keys():
            sig_dict = {'EventNumber': event, 'NumParticlesInEvent': n_part,
                        "PerfectReco": 0, "AllParticles": 0, "NoneIso": 0, "PartReco": 0, "NotFound": 0,
                        "NumBkgParticles_noniso": -999}
            tc = tc_dict[tc_key]
            sig_dict["SigMatch"] = 0
            if ref_signal:
                labels = tc['labels']
                mothers = [label[3:] for label in labels if 'c' == label[0]]
                node_keys = tc['node_keys']
                daughters = [label.split(':')[1] for label in labels if int(label.split(':')[0][1:]) in node_keys]
                if match_decays(daughters, ref_signal[0]['daughters']) or match_decays(daughters,
                                                                                       ref_signal[1]['daughters']):
                    check_mothers1 = True
                    check_mothers2 = True
                    for i in range(len(ref_signal[0]['mothers'])):
                        if ref_signal[0]['mothers'][i] not in mothers:
                            check_mothers1 = False
                        if ref_signal[1]['mothers'][i] not in mothers:
                            check_mothers2 = False
                    sig_dict["SigMatch"] = int(check_mothers1 or check_mothers2)

            sig_dict["NumSignalParticles"] = len(tc['node_keys'])

            if tc_key in rc_dict.keys():
                perfect_sig_reco = int(
                    rc_dict[tc_key]['node_keys'] == tc['node_keys']
                    and rc_dict[tc_key]['LCA_values'] == tc['LCA_values']
                )
            else:
                perfect_sig_reco = 0
            perfect_evt_reco *= perfect_sig_reco

            for rc in rc_dict.values():
                true_in_reco = np.sum(np.isin(tc['node_keys'], rc['node_keys'])) / len(tc['node_keys'])
                if rc['node_keys'] == tc['node_keys']:
                    sig_dict["AllParticles"] = 1
                    if rc['LCA_values'] == tc['LCA_values']:
                        sig_dict["PerfectReco"] = 1
                    sig_dict["NoneIso"] = sig_dict["PartReco"] = 0
                    sig_dict = get_pred_ft(sig_dict, graph, rc, ft_des)
                    sig_dict = get_asso_frag(sig_dict, graph, rc)
                    break
                elif true_in_reco == 1 and len(rc['node_keys']) > len(tc['node_keys']):
                    sig_dict["NoneIso"] = 1  # background tracks in signal
                    sig_dict["PartReco"] = 0
                    sig_dict = get_pred_ft(sig_dict, graph, rc, ft_des)
                    sig_dict = get_asso_frag(sig_dict, graph, rc)
                    break
                elif 0.2 <= true_in_reco < 1:
                    sig_dict["PartReco"] = 1  # FT decision can not be trusted
                    sig_dict["NumBkgParticles_noniso"] = len(rc['node_keys']) - len(tc['node_keys'])
                    sig_dict = get_pred_ft(sig_dict, graph, rc, ft_des)
                    sig_dict = get_asso_frag(sig_dict, graph, rc)
                    break
                else:
                    sig_dict["final_pid"] = sig_dict["final_b_score"] = sig_dict["final_bbar_score"] = ""
                    sig_dict["ft_b_score"] = sig_dict["ft_bbar_score"] = 0

            if sig_dict["AllParticles"] == 0 and sig_dict["NoneIso"] == 0 and sig_dict["PartReco"] == 0:
                sig_dict["NotFound"] = 1

            # Get origin B id
            indices = [particle_keys.index(x) for x in tc['node_keys']]
            signal_LCA_id = true_LCA[true_LCA['senders'].isin(indices) | true_LCA['receivers'].isin(indices)]["LCA_id"]
            values, counts = np.unique(signal_LCA_id, return_counts=True)
            max_indices = np.where(counts == counts.max())[0]
            if len(max_indices) == 1:
                sig_dict["B_id"] = values[max_indices[0]]
            else:
                candidate_lca_ids = values[max_indices]
                candidates_df = true_LCA[true_LCA['LCA_id'].isin(candidate_lca_ids)]
                max_chain_per_lca = candidates_df.groupby('LCA_id')['TrueFullChainLCA'].max()
                sig_dict["B_id"] = max_chain_per_lca.idxmax()

            sig_df = sig_df._append(sig_dict, ignore_index=True)

        # temp stuff
        evt_df = evt_df._append({'EventNumber': event,
                                 'NumParticlesInEvent': n_part,
                                 'NumParticlesFromHeavyHadronInEvent': n_part_heavy_h,
                                 'NumBackgroundParticlesInEvent': n_bkg_part,
                                 'NumSelectedParticlesInEvent': n_sel_part,
                                 'NumSelectedParticlesFromHeavyHadronInEvent': n_sel_heavy_h,
                                 'NumSelectedBackgroundParticlesInEvent': n_sel_bkg_part,
                                 'NumTruthClustersGen1': t_nclust_order[0],
                                 'NumTruthClustersGen2': t_nclust_order[1],
                                 'NumTruthClustersGen3': t_nclust_order[2],
                                 'NumTruthClustersGen4': t_nclust_order[3],
                                 'NumRecoClustersGen1': r_nclust_order[0],
                                 'NumRecoClustersGen2': r_nclust_order[1],
                                 'NumRecoClustersGen3': r_nclust_order[2],
                                 'NumRecoClustersGen4': r_nclust_order[3],
                                 'MaxTruthFullChainDepthInEvent': max_chain_depth,
                                 'PerfectEventReconstruction': perfect_evt_reco,
                                 'NumTrueSignalsInEvent': len(tc_dict.keys()),
                                 'NumRecoSignalsInEvent': len(rc_dict.keys()),
                                 },
                                ignore_index=True)
    return sig_df, evt_df
