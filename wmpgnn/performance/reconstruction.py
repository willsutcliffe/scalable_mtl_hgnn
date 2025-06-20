from particle import Particle
from numpy import intersect1d
from networkx.drawing.nx_agraph import graphviz_layout
import pandas as pd
import numpy as np
import networkx as nx
# This file contains important reconstruction methods from the original DFEI paper.

def particle_name(id_):
    """
    Maps a particle ID to its corresponding human-readable name.

    This function provides specific names for a few predefined particle IDs
    and falls back to a general particle naming utility (e.g., from the `particle`
    package) for other IDs.

    Args:
        id_ : int
            The integer ID of the particle (e.g., PDG ID).

    Returns:
        str
            The name of the particle.

    Examples:
        >>> particle_name(0)
        'ghost'
        >>> particle_name(10413)
        'D1(2420)+'
        >>> particle_name(211) # Assuming Particle.from_pdgid(211).name returns 'pi+'
        'pi+'
    """
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

def make_decay_dict(decay):
    """
    Creates a frequency dictionary from a list of particles in a decay.

    This dictionary counts the occurrences of each unique particle in the
    provided `decay` list.

    Args:
        decay : list
            A list of particles (e.g., strings representing particle names or IDs)
            in a decay chain.

    Returns:
        dict
            A dictionary where keys are unique particles from the `decay` list
            and values are their respective counts.

    Examples:
        >>> make_decay_dict(['pi+', 'K-', 'pi+', 'gamma'])
        {'pi+': 2, 'K-': 1, 'gamma': 1}
    """
    decay_dict ={}
    for particle in decay:
        if particle not in decay_dict.keys():
            decay_dict[particle]=1
        else:
            decay_dict[particle]+=1
    return decay_dict

def match_decays(decay1, decay2):
    """
    Compares two decay lists to determine if they represent the same set of
    particles with the same multiplicities, regardless of order.

    This function converts each decay list into a frequency dictionary using
    `make_decay_dict` and then compares these dictionaries for equivalence.

    Args:
        decay1 : list
            The first list of particles representing a decay.
        decay2 : list
            The second list of particles representing a decay.

    Returns:
        bool
            True if the two decay lists contain the same particles with the same
            counts; False otherwise.

    Examples:
        >>> match_decays(['pi+', 'K-'], ['K-', 'pi+'])
        True
        >>> match_decays(['pi+', 'pi-'], ['pi+', 'pi+', 'K-'])
        False
    """
    decay_dict1 = make_decay_dict(decay1)
    decay_dict2 = make_decay_dict(decay2)
    if len(decay_dict1.keys()) != len(decay_dict2.keys()):
        return False
    decay_dict2_keys = decay_dict2.keys()
    for key in decay_dict1.keys():
        if key not in decay_dict2_keys:
            return False
        elif decay_dict1[key] != decay_dict2[key]:
            return False
    return True

def flatten(t):
    """
    Flattens a list of lists into a single, one-dimensional list.

    Args:
        t : list
            A list containing sublists.

    Returns:
        list
            A new list containing all elements from the sublists in `t`,
            in the order they appeared.
    """
    return [item for sublist in t for item in sublist]

def compute_LCA(anc1, anc2, max_depth):
    """
    Computes the Lowest Common Ancestor (LCA) value between two lists of ancestors.

    The LCA value represents the "depth" of the lowest common ancestor in a
    hierarchical structure, relative to the maximum possible depth. A higher
    LCA value indicates a common ancestor further down the decay chain (closer
    to the final state particles).

    Args:
        anc1 : list
            A list of ancestor indices for the first particle, ordered from
            the "root" (earliest ancestor) to the particle itself.
        anc2 : list
            A list of ancestor indices for the second particle, ordered from
            the "root" (earliest ancestor) to the particle itself.
        max_depth : int
            The maximum possible depth of a decay chain in the event. This is used
            to normalize the LCA value.

    Returns:
        int
            The LCA value, calculated as `max_depth - lowest_common_ancestor_generation`.
            Returns 0 if either ancestor list is empty or if no common ancestors are found.

    Notes:
        - The `intersect1d` function is assumed to be imported (e.g., from NumPy).
        - The ancestor indices are expected to be ordered such that reversing
          `common_ancestors` places the lowest common ancestor at the end of the list.
        - The `lowest_common_ancestor_generation` is the index of the LCA within
          the longer of the two ancestor lists, representing its depth.
    """

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
    """
    Reconstructs decay chains from a Lowest Common Ancestor (LCA) matrix
    and associated particle information. It identifies clusters of connected
    particles and their hierarchical relationships, which can be visualized
    as a decay tree.

    This function processes the LCA matrix to build a graph representation
    of decay chains. It can handle both reconstructed data and truth-level
    simulations, and optionally plots the reconstructed trees.

    Args:
        triang_LCA_matrix : pandas.DataFrame or numpy.ndarray
            A matrix containing LCA information for pairs of particles. Expected
            columns/structure are 'senders', 'receivers', and 'LCA_dec'
            (LCA decay depth). For `truth_level_simulation=1`, it also expects
            'LCA_id_label' and 'TrueFullChainLCA'.
        particle_keys : list
            A list of unique identifiers (keys) for the particles in the event.
            These keys are used to reference particles in the LCA matrix.
        ax : matplotlib.axes.Axes, optional
            A Matplotlib Axes object to draw the reconstructed decay tree on.
            If 0 (default), no plot is generated.
        particle_ids : list, optional
            A list of particle IDs (e.g., PDG IDs or names) corresponding to
            `particle_keys`. Used for more descriptive labels in truth-level
            simulations. If empty, generic 'k' + key labels are used. Defaults
            to an empty list.
        truth_level_simulation : int, optional
            A flag indicating whether the input data is from a truth-level
            simulation.
            - If 1, the function extracts additional truth-level information
              like `LCA_id_label` and `TrueFullChainLCA` for cluster labeling
              and max full chain depth calculation.
            - If 0 (default), it treats the data as reconstructed and uses
              generic 'reco_c' labels for composite particles.

    Returns:
        tuple
            A tuple containing:
            - cluster_dict : dict
                A dictionary where keys are the smallest particle key in each
                reconstructed decay chain, and values are dictionaries containing
                'node_keys' (list of particle keys in the cluster), 'LCA_values'
                (list of concatenated LCA values for pairs within the cluster),
                and 'labels' (labels for all nodes, including composite particles).
            - num_clusters_per_order : dict
                A dictionary tracking the number of composite clusters found at
                each decay order (depth). Keys are decay orders (0, 1, 2, 3),
                and values are the counts.
            - max_full_chain_depth_in_event : int
                The maximum true full chain LCA depth found in the event.
                Only relevant if `truth_level_simulation` is 1; otherwise, -1.

    Notes
    -----
    - This function assumes the existence of `compute_LCA`, `particle_name`
      (if `particle_ids` is used), and `flatten` (if used to process
      `truth_cluster_dict` or `reco_cluster_dict` in calling functions).
      These helper functions are not defined within `reconstruct_decay`.
    - The `_append` method for pandas DataFrames used in the original code
      is deprecated. For newer pandas versions, `pd.concat` should be used
      for appending rows to a DataFrame. The provided docstring reflects
      the original code's behavior.
    """
    num_clusters_per_order = {}
    for order_ in range(4):
        num_clusters_per_order[order_] = 0

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

        try:
            pos = graphviz_layout(G, prog='dot')
            nx.draw(G, pos, with_labels=False,
                    node_color=filtered_node_colors, node_size=1300, ax=ax)
            label_options = {"ec": "k", "fc": "white", "alpha": 0.7}
            nx.draw_networkx_labels(G, pos, font_size=14,
                                    bbox=label_options, ax=ax)
        except ImportError:
            print("graphviz not installed, cannot plot decay tree. Please install graphviz to visualize the decay tree.")

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
            'node_keys': clustered_keys[ic], 'LCA_values': clustered_concatenated_LCA_values[ic],
        'labels' : labels}

    return cluster_dict, num_clusters_per_order, max_full_chain_depth_in_event

