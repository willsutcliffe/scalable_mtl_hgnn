import torch
from torch_scatter import scatter_add

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

def acc_n_class(pred, label, n_class=4):
    correct = 0
    correct_class = {i : 0 for i in range(n_class)}
    pred_argmax = torch.argmax(pred, dim=1)
    
    pred_class = {i : (pred_argmax == i).sum() for i in range(n_class)}
    true_class = {i : (label == i).sum() for i in range(n_class)}
    
    if len(pred) != len(label):
        print("something goes wrong in acc_n_class")
        print(len(pred), len(label))
    else:
        for i in range(n_class):
            correct_class[i] = torch.sum(pred_argmax[label == i] == label[label == i])

    correct_preds = torch.Tensor([correct_class[i] for i in range(n_class)])
    all_preds = torch.Tensor(tuple(pred_class[i] for i in range(n_class)))
    all_label = torch.Tensor(tuple(true_class[i] for i in range(n_class)))

    acc = torch.div(correct_preds, all_label)

    return acc
    

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

    pred_argmax = torch.argmax(pred, dim=1)


    pred_class1 = (pred_argmax == 0).sum()
    pred_class2 = (pred_argmax == 1).sum()
    pred_class3 = (pred_argmax == 2).sum()
    pred_class4 = (pred_argmax == 3).sum()

    true_class1 = (label == 0).sum()
    true_class2 = (label == 1).sum()
    true_class3 = (label == 2).sum()
    true_class4 = (label == 3).sum()

    if len(pred) != len(label):
        print("something goes wrong in acc_four_class")
        print(len(pred), len(label))

    else:
        correct_class1 = torch.sum(pred_argmax[label == 0] == label[label == 0])
        correct_class2 = torch.sum(pred_argmax[label == 1] == label[label == 1])
        correct_class3 = torch.sum(pred_argmax[label == 2] == label[label == 2])
        correct_class4 = torch.sum(pred_argmax[label == 3] == label[label == 3])

    correct_preds = torch.Tensor([correct_class1, correct_class2, correct_class3, correct_class4])
    all_preds = torch.Tensor((pred_class1, pred_class2, pred_class3, pred_class4))
    all_label = torch.Tensor((true_class1, true_class2, true_class3, true_class4))

    acc = torch.div(correct_preds, all_label)

    return acc

def weight_n_class(dataset,hetero=False,n_class=4):
    num_sample = 0
    true_class = {i: 0 for i in range(n_class)}
    
    for tdata in dataset:
        if hetero:
            y = tdata[('tracks','to','tracks')].y
        else:
            y = tdata.y
        for i in range(n_class):
            true_class[i] += (y.argmax(dim=1) == i).sum()
        num_sample += len(y)
    
    weight_class = {i: num_sample / (n_class * true_class[i]) for i in range(n_class)}
    weight = torch.stack(tuple(weight_class[i] for i in range(n_class)))

    print(weight)
    return weight

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
