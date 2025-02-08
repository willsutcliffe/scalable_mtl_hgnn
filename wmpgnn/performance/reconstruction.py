from particle import Particle
from numpy import intersect1d
from networkx.drawing.nx_agraph import graphviz_layout
import pandas as pd
import numpy as np
import networkx as nx
# This file contains important reconstruction methods from the original DFEI paper.

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
    else:
        return Particle.from_pdgid(id_).name

def flatten(t):
    return [item for sublist in t for item in sublist]

def compute_LCA(anc1, anc2, max_depth):

    if (anc1 == []) or (anc2 == []):
        return 0

    common_ancestors = intersect1d(anc1, anc2).tolist()
    # IMPORTANT!!: the order of the ancestor indices reconstructed by this algorithm is the opposite of the one used in simulation, so the order must be reversed in this case.
    common_ancestors.reverse()

    if (common_ancestors == []):
        return 0

    lowest_common_ancestor = common_ancestors[-1]

    if (len(anc1) >= len(anc2)):
        max_length = anc1
    else:
        max_length = anc2
    lowest_common_ancestor_generation = max_length.index(
        lowest_common_ancestor)

    return max_depth - lowest_common_ancestor_generation


def reconstruct_decay(triang_LCA_matrix, particle_keys, ax=0, particle_ids=[], truth_level_simulation=0):
    '''
    Function used to reconstruct the decay chain
    '''
    # Nmax = np.max(np.unique( np.array( list(triang_LCA_matrix['senders']) + list(triang_LCA_matrix['receivers']))))
    num_clusters_per_order = {}
    for order_ in range(4):
        num_clusters_per_order[order_] = 0
    # particle_keys = list(range(0,Nmax+1))
    if particle_ids == []:
        labels = list(map(lambda x: 'k' + str(x), particle_keys))
    else:
        labels = list(map(lambda x, y: 'k' + str(x) + ':' +
                                       y, particle_keys, particle_ids))
    node_colors = []
    for l in labels:
        node_colors.append('#3e5948')

    max_full_chain_depth_in_event = -1

    # Create the global LCA matrix for the event, and remove null connections
    current_LCA_matrix = pd.DataFrame(triang_LCA_matrix, columns=[
        'senders', 'receivers', 'LCA_dec'])
    current_LCA_matrix = current_LCA_matrix[current_LCA_matrix['LCA_dec'] > 0]

    # Check against empty events
    if current_LCA_matrix.empty:
        print('No particles found.')
        print(truth_level_simulation)
        return {}, num_clusters_per_order, max_full_chain_depth_in_event

    # Create a dictionary to store the true ID of the ancestors

    if truth_level_simulation:
        cluster_label_dict = pd.DataFrame(triang_LCA_matrix, columns=[
            'senders', 'receivers', 'LCA_id_label', 'TrueFullChainLCA'])
        cluster_label_dict.set_index(['senders', 'receivers'], inplace=True)
        max_full_chain_depth_in_event = max(
            cluster_label_dict['TrueFullChainLCA'].values)

    # Define an auxiliary matrix to later identify connected clusters
    num_nodes = len(particle_keys)

    clustering_adjacency_matrix = np.zeros((num_nodes, num_nodes))

    # Get the maximum of the LCA matrix
    max_depth = np.max(current_LCA_matrix['LCA_dec'])

    adj_links_list = []

    composite_counter = num_nodes - 1

    for order in range(max_depth):

        # print('order',order)

        LCA_matrix_subset = current_LCA_matrix[current_LCA_matrix['LCA_dec'] == 1]
        if LCA_matrix_subset.empty == False:

            # Reset the clustering adjacency matrix, and set it up to study the next LCA order
            clustering_adjacency_matrix = np.zeros(
                (composite_counter + 1, composite_counter + 1))

            for ie in range(LCA_matrix_subset.shape[0]):
                clustering_adjacency_matrix[LCA_matrix_subset.iloc[ie]
                ['senders']][LCA_matrix_subset.iloc[ie]['receivers']] = 1
                clustering_adjacency_matrix[LCA_matrix_subset.iloc[ie]
                ['receivers']][LCA_matrix_subset.iloc[ie]['senders']] = 1
            nx_graph = nx.from_numpy_array(clustering_adjacency_matrix)
            connected_components = [
                list(x) for x in nx.connected_components(nx_graph) if len(x) > 1]

            # Inspect the separate clusters iteratively
            cl_counter = 0
            for indices_in_cluster in connected_components:

                # print('cluster counter',cl_counter)
                cl_counter += 1

                # Label the new cluster
                composite_counter += 1
                num_clusters_per_order[order] += 1
                if truth_level_simulation:
                    proxy_link = LCA_matrix_subset[(LCA_matrix_subset['senders'].isin(indices_in_cluster)) & (
                        LCA_matrix_subset['receivers'].isin(indices_in_cluster))].iloc[0]
                    labels.append('c' + str(composite_counter - num_nodes + 1) + ':' + cluster_label_dict.loc[(
                        proxy_link['senders'], proxy_link['receivers'])]['LCA_id_label'])
                else:
                    labels.append(
                        'reco_c' + str(composite_counter - num_nodes + 1))
                node_colors.append('#91b39d')

                # Pass the information to the reconstructed adjacency matrix
                for ind in indices_in_cluster:
                    new_df = pd.DataFrame({'senders': [ind],
                                           'receivers': [composite_counter],
                                           'link': [1]})
                    adj_links_list.append(new_df)

                # If there was any connection between the other nodes and any of the particles in the new cluster, connect those nodes to the new cluster as appropriate
                for sender in range(composite_counter):
                    if sender not in indices_in_cluster:
                        proxy_links = current_LCA_matrix[
                            ((current_LCA_matrix['senders'] == sender) & (current_LCA_matrix['receivers'].isin(
                                indices_in_cluster))) | ((current_LCA_matrix['senders'].isin(indices_in_cluster)) & (
                                        current_LCA_matrix['receivers'] == sender))]
                        if proxy_links.empty == False:
                            new_LCA_matrix_df = pd.DataFrame({
                                'senders': [sender],
                                'receivers': [composite_counter],
                                'LCA_dec': [max(proxy_links['LCA_dec'])]
                            })
                            LCA_matrix_list = []
                            LCA_matrix_list.append(current_LCA_matrix)
                            LCA_matrix_list.append(new_LCA_matrix_df)
                            current_LCA_matrix = pd.concat(
                                LCA_matrix_list, ignore_index=True)
                            if truth_level_simulation:
                                cluster_label_dict.loc[(sender, composite_counter), 'LCA_id_label'] = \
                                cluster_label_dict.loc[(
                                    proxy_links['senders'].iloc[0], proxy_links['receivers'].iloc[0]), 'LCA_id_label']

                # Remove connections with the nodes inside the new cluster
                current_LCA_matrix = current_LCA_matrix[(current_LCA_matrix['senders'].isin(
                    indices_in_cluster) == False) & (current_LCA_matrix['receivers'].isin(indices_in_cluster) == False)]

        current_LCA_matrix['LCA_dec'] = current_LCA_matrix['LCA_dec'] - 1
        current_LCA_matrix = current_LCA_matrix[current_LCA_matrix['LCA_dec'] > 0]

    if (adj_links_list):
        adj_links = pd.concat(adj_links_list, ignore_index=True)

    # Plot the tree
    if ax != 0:
        G = nx.DiGraph()

        adj_senders = adj_links['senders'].to_list()
        adj_receivers = adj_links['receivers'].to_list()
        filtered_node_colors = []
        for i in range(len(labels)):
            if i in adj_senders or i in adj_receivers:
                G.add_node(labels[i])
                filtered_node_colors.append(node_colors[i])

        for ie in range(adj_links.shape[0]):
            edge = adj_links.iloc[ie]
            G.add_edge(labels[edge['receivers']], labels[edge['senders']])

        pos = graphviz_layout(G, prog='dot')
        nx.draw(G, pos, with_labels=False,
                node_color=filtered_node_colors, node_size=1300, ax=ax)
        label_options = {"ec": "k", "fc": "white", "alpha": 0.7}
        nx.draw_networkx_labels(G, pos, font_size=14,
                                bbox=label_options, ax=ax)

    # Compute information per separated decay chain

    final_adjacency_matrix = np.zeros(
        (composite_counter + 1, composite_counter + 1))
    for ie in range(adj_links.shape[0]):
        final_adjacency_matrix[adj_links.iloc[ie]
        ['senders']][adj_links.iloc[ie]['receivers']] = 1
        final_adjacency_matrix[adj_links.iloc[ie]
        ['receivers']][adj_links.iloc[ie]['senders']] = 1
    nx_graph = nx.from_numpy_array(final_adjacency_matrix)
    connected_components = [
        list(x) for x in nx.connected_components(nx_graph) if len(x) > 1]

    clustered_keys = []
    clustered_concatenated_LCA_values = []

    for nodes_in_cluster in connected_components:

        # Identify the keys of the final particles in the cluster, and list them in ascending order
        index_from_key_dict = {}
        for node in nodes_in_cluster:
            if node < num_nodes:
                index_from_key_dict[particle_keys[node]] = node
        ordered_keys_in_cluster = list(index_from_key_dict.keys())
        ordered_keys_in_cluster.sort()
        clustered_keys.append(ordered_keys_in_cluster)

        # Identify the list of ancestors for each final state particle
        ancestor_lists_in_cluster = []
        for k in ordered_keys_in_cluster:
            node_index = index_from_key_dict[k]
            ancestor_list = []
            current_link = adj_links[adj_links['senders'] == node_index]
            while current_link.empty == False:
                current_receiver = current_link.iloc[0]['receivers']
                ancestor_list.append(current_receiver)
                current_link = adj_links[adj_links['senders']
                                         == current_receiver]
            ancestor_list.reverse()
            ancestor_lists_in_cluster.append(ancestor_list)
        max_decay_length = max([len(x) for x in ancestor_lists_in_cluster])

        # Compute the LCA values and concatenate them following a given order
        concatenated_LCA_values_in_cluster = []
        for in1 in range(len(ordered_keys_in_cluster)):
            for in2 in range(len(ordered_keys_in_cluster)):
                if in1 < in2:
                    concatenated_LCA_values_in_cluster.append(compute_LCA(
                        ancestor_lists_in_cluster[in1], ancestor_lists_in_cluster[in2], max_decay_length))
        clustered_concatenated_LCA_values.append(
            concatenated_LCA_values_in_cluster)

    # Store the cluster information in a dictionary, with entries given by the smallest key value
    cluster_dict = {}
    for ic in range(len(connected_components)):
        cluster_dict[clustered_keys[ic][0]] = {
            'node_keys': clustered_keys[ic], 'LCA_values': clustered_concatenated_LCA_values[ic]}

    return cluster_dict, num_clusters_per_order, max_full_chain_depth_in_event

    return cluster_dict, num_clusters_per_order, max_full_chain_depth_in_event