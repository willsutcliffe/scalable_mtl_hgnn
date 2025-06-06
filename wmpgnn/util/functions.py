import torch
from torch_scatter import scatter_add

from tqdm import tqdm
from itertools import islice


def hetero_positive_edge_weight(loader):
    """
    Computes the positive class weighting factor for edges in a heterogeneous graph
    for binary classification (positive class = label 0).

    Parameters
    ----------
    loader : DataLoader
        A DataLoader yielding heterogeneous graphs with edge attributes
        `y` for edge labels under key ('tracks', 'to', 'tracks').

    Returns
    -------
    float
        The ratio `total_edges / (2 * num_positive_edges)`, used for loss weighting.
    """
    sum_edges = 0
    sum_pos = 0
    for data in loader:
        sum_edges += data[('tracks','to','tracks')].edges.shape[0]
        sum_pos  += torch.sum(data[('tracks','to','tracks')].y[:,0]==0).item()
    return sum_edges/(2*sum_pos)

def hetero_positive_node_weight(loader):
    """
    Computes the positive class weighting factor for nodes in a heterogeneous graph.
    A node is considered positive if any incoming edge has a positive label (nonzero).

    Parameters
    ----------
    loader : DataLoader
        A DataLoader yielding heterogeneous graphs with edge labels `y` and
        edge_index under key ('tracks', 'to', 'tracks').

    Returns
    -------
    float
        The ratio `total_nodes / (2 * num_positive_nodes)`, used for loss weighting.
    """
    sum_nodes = 0
    sum_pos = 0
    for data in loader:
        num_nodes=data['tracks'].x.shape[0]
        node_sum = scatter_add(data[('tracks','to','tracks')].y, data[('tracks','to','tracks')].edge_index[0],dim=0)
        ynodes = (1.*(torch.sum(node_sum[:,1:],1)>0)).unsqueeze(1)
        sum_nodes += num_nodes
        sum_pos  += torch.sum(ynodes==1).item()
    return sum_nodes/(2*sum_pos)


def positive_edge_weight(loader):
    """
    Computes the positive class weighting factor for edges in a homogeneous graph
    for binary classification (positive class = label 0).

    Parameters
    ----------
    loader : DataLoader
        A DataLoader yielding graphs with edge labels `y`.

    Returns
    -------
    float
        The ratio `total_edges / (2 * num_positive_edges)`, used for loss weighting.
    """
    sum_edges = 0
    sum_pos = 0
    for data in loader:
        sum_edges += data.edges.shape[0]
        sum_pos  += torch.sum(data.y[:,0]==0).item()
    return sum_edges/(2*sum_pos)

def positive_node_weight(loader):
    """
    Computes the positive class weighting factor for nodes in a homogeneous graph.
    A node is considered positive if any of its incoming edges are positive
    (nonzero labels).

    Parameters
    ----------
    loader : DataLoader
        A DataLoader yielding graphs with node features and edge labels `y`.

    Returns
    -------
    float
        The ratio `total_nodes / (2 * num_positive_nodes)`, used for loss weighting.
    """
    sum_nodes = 0
    sum_pos = 0
    for data in loader:
        num_nodes=data.nodes.shape[0]
        node_sum = scatter_add(data.y,data.senders,dim=0)
        ynodes = (1.*(torch.sum(node_sum[:,1:],1)>0)).unsqueeze(1)
        sum_nodes += num_nodes
        sum_pos  += torch.sum(ynodes==1).item()
    return sum_nodes/(2*sum_pos)


def acc_four_class(pred, label):
    """
    Computes the per-class accuracy for a 4-class classification task.

    Parameters
    ----------
    pred : torch.Tensor
        The predicted logits or probabilities, shape [N, 4].
    label : torch.Tensor
        The true class labels, shape [N].

    Returns
    -------
    torch.Tensor
        A tensor of shape [4] containing the accuracy for each class.
    """
    correct = 0
    correct_class1 = 0
    correct_class2 = 0
    correct_class3 = 0
    correct_class4 = 0
    class0_selbool = label == 0
    class1_selbool = label == 1
    class2_selbool = label == 2
    class3_selbool = label == 3
    

    pred_argmax = torch.argmax(pred, dim=1)

    res = {}
    for i in range(4):
        classi_selbool = label == i

        res[f"LCA_class{i}_num"] = torch.sum(classi_selbool).float().item()
        if res[f"LCA_class{i}_num"] == 0:
            res[f"LCA_class{i}_pred_class0"] = 0.
            res[f"LCA_class{i}_pred_class1"] = 0.
            res[f"LCA_class{i}_pred_class2"] = 0.
            res[f"LCA_class{i}_pred_class3"] = 0.
        else:
            res[f"LCA_class{i}_pred_class0"] = torch.sum(pred_argmax[classi_selbool] == 0).item() / res[f"LCA_class{i}_num"]
            res[f"LCA_class{i}_pred_class1"] = torch.sum(pred_argmax[classi_selbool] == 1).item() / res[f"LCA_class{i}_num"]
            res[f"LCA_class{i}_pred_class2"] = torch.sum(pred_argmax[classi_selbool] == 2).item() / res[f"LCA_class{i}_num"]
            res[f"LCA_class{i}_pred_class3"] = torch.sum(pred_argmax[classi_selbool] == 3).item() / res[f"LCA_class{i}_num"]
    return res


def weight_four_class(dataset,hetero=False):
    """
    Computes inverse-frequency class weights for a 4-class classification task.

    Parameters
    ----------
    dataset : iterable
        A dataset of graph objects with multi-class labels.
    hetero : bool, optional
        If True, assumes heterogeneous graph format and accesses labels via
        `('tracks', 'to', 'tracks')`.

    Returns
    -------
    torch.Tensor
        A tensor of shape [4] containing the class weights.
    """
    true_class1 = 0
    true_class2 = 0
    true_class3 = 0
    true_class4 = 0
    num_sample = 0

    for tdata in dataset:
        if hetero:
            y = tdata[('tracks','to','tracks')].y
        else:
            y = tdata.y
        true_class1 += (y.argmax(dim=1) == 0).sum()
        true_class2 += (y.argmax(dim=1) == 1).sum()
        true_class3 += (y.argmax(dim=1) == 2).sum()
        true_class4 += (y.argmax(dim=1) == 3).sum()
        num_sample += len(y)


    weight_class1 = num_sample / (4 * true_class1)
    weight_class2 = num_sample / (4 * true_class2)
    weight_class3 = num_sample / (4 * true_class3)
    weight_class4 = num_sample / (4 * true_class4)
    weight = torch.stack((weight_class1, weight_class2, weight_class3, weight_class4))

    print(weight)
    return weight


def get_hetero_weight(loader, node_weight=True, edge_weight=True, LCA_weight=True, frag_weight=True, ft_weight=True):
    true_class = torch.zeros(4)
    num_sample = 0

    sum_nodes_neg = 0
    sum_nodes_pos = 0

    sum_edges_neg = 0
    sum_edges_pos = 0

    pos_frag = 0
    neg_frag = 0

    ft_counts = torch.zeros(3)

    for data in tqdm(islice(loader, 1000), total=1000): # not sure why it takes so long need to be optimized to save at least some time
        if node_weight:
            node_sum = scatter_add(data[('tracks','to','tracks')].y, data[('tracks','to','tracks')].edge_index[0], dim=0)
            ynodes = (torch.sum(node_sum[:,1:],1) > 0).unsqueeze(1)
            sum_nodes_pos  += torch.sum(ynodes==1).item()
            sum_nodes_neg  += torch.sum(ynodes==0).item()

        if edge_weight:
            sum_edges_pos  += torch.sum(data[('tracks','to','tracks')].y[:,0]==0).item()
            sum_edges_neg  += torch.sum(data[('tracks','to','tracks')].y[:,0]==1).item()

        if LCA_weight:
            y = data[('tracks','to','tracks')].y
            classes = y.argmax(dim=1)
            binc = torch.bincount(classes, minlength=4) 
            true_class += binc
            num_sample += classes.size(0)

        if frag_weight:
            y = data['tracks'].frag
            pos_frag += torch.sum(data['tracks'].frag.squeeze() != 0)
            neg_frag += y.shape[0] - torch.sum(y)
        
        if ft_weight:
            y = data['tracks'].ft + 1
            ft_counts += torch.bincount(y)
        
    weight_class = num_sample / (4 * true_class)
    weight_nodes = torch.tensor(sum_nodes_neg/sum_nodes_pos)
    weight_edges = torch.tensor(sum_edges_neg/sum_edges_pos)
    weight_frag = neg_frag / pos_frag
    weight_ft = torch.sum(ft_counts) / (3 * ft_counts)


    pos_weight ={"t_nodes": weight_nodes, "tt_edges": weight_edges, "LCA": weight_class, "frag": weight_frag, "FT": weight_ft}
    return pos_weight

def init_plot_style():
    """
    Initializes and returns a dictionary of matplotlib RC parameters for
    producing clean, publication-quality plots.

    Returns
    -------
    dict
        Dictionary of matplotlib style parameters.
    """
    my_rc_params = {
        "xtick.direction": "in",
        "xtick.major.size": 8.0,
        "xtick.minor.size": 4.0,
        "xtick.minor.visible": True,
        "xtick.major.width": 1.2,
        "xtick.minor.width": 0.9,
        "ytick.direction": "in",
        "ytick.major.size": 8.0,
        "ytick.minor.size": 4.0,
        "ytick.minor.visible": True,
        "ytick.major.width": 1.2,
        "ytick.minor.width": 0.9,
        "errorbar.capsize": 2,
        "axes.linewidth": 1.2,
        # "font.familiy": "serif",
        "font.size": 14,
        "axes.grid": False,
        "ytick.right": True,
        "xtick.top": True
    }
    return(my_rc_params)
