import torch
import numpy as np
from torch_scatter import scatter_add
from datetime import datetime
from scipy.stats import ks_2samp


def neutrals_hetero_positive_edge_weight(loader):
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
        #print("A:\t",data[('tracks','to','tracks')].edges.shape[0])
        #print("B:\t",torch.sum(data[('tracks','to','tracks')].y[:,0]==0).item())
        sum_edges += data[('chargedtree','to','neutrals')].edges.shape[0]
        sum_pos  += torch.sum(data[('chargedtree','to','neutrals')].y[:,0]==0).item()
    return sum_edges/(2*sum_pos)

def neutrals_hetero_positive_node_weight(loader):
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
        num_nodes=data['neutrals'].x.shape[0]
        print("C:\t",num_nodes)
        #out = data.edges.new_zeros(num_nodes, 4)
        node_sum = scatter_add(data[('chargedtree','to','neutrals')].y, data[('chargedtree','to','neutrals')].edge_index[0],dim=0)
        ynodes = (1.*(torch.sum(node_sum[:,1:],1)>0)).unsqueeze(1)
        sum_nodes += num_nodes
        sum_pos  += torch.sum(ynodes==1).item()
    return sum_nodes/(2*sum_pos)

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
        #print("A:\t",data[('tracks','to','tracks')].edges.shape[0])
        #print("B:\t",torch.sum(data[('tracks','to','tracks')].y[:,0]==0).item())
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
        print("C:\t",num_nodes)
        #out = data.edges.new_zeros(num_nodes, 4)
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
        print("D:\t",data.edges.shape[0])
        print("E:\t",torch.sum(data.y[:,0]==0).item())
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
        #out = data.edges.new_zeros(num_nodes, 4)
        node_sum = scatter_add(data.y,data.senders,dim=0)
        ynodes = (1.*(torch.sum(node_sum[:,1:],1)>0)).unsqueeze(1)
        sum_nodes += num_nodes
        sum_pos  += torch.sum(ynodes==1).item()
    return sum_nodes/(2*sum_pos)

def compute_efficiency_error(num, den):
    """
    Compute the statistical error on efficiency.

    Parameters
    ----------
    num : number of successes (e.g. true positives)
    den : total number of trials (e.g. total positives)

    Returns
    -------
    Efficiency error calculated as sqrt(num * (den - num) / den^3).
    """
    a = den - num
    return torch.sqrt(num * a / ((num + a) ** 3))


def eff_binary(pred, label):
    """
    Compute binary signal efficiency (recall) for class 1.

    Efficiency = True Positives (class 1) / Total actual positives (class 1).

    Parameters
    ----------
    pred : torch.Tensor
        Predicted binary labels (0 or 1).
    label : torch.Tensor
        True binary labels (0 or 1).

    Returns
    -------
    torch.Tensor
        Signal efficiency for class 1.
    """
    true_positives = (pred * label).sum().float()  # True Positives for class 1
    total_positives = label.sum().float()          # Total actual positives for class 1

    if total_positives > 0:
        eff = true_positives / total_positives
    else:
        eff = torch.tensor(0.0)

    return eff


def rej_binary(pred, label):
    """
    Compute binary background rejection for class 1.

    Rejection = True Negatives / (True Negatives + False Positives).

    Parameters
    ----------
    pred : torch.Tensor
        Predicted binary labels (0 or 1).
    label : torch.Tensor
        True binary labels (0 or 1).

    Returns
    -------
    torch.Tensor
        Background rejection for class 1.
    """
    true_negatives = ((pred == 0) & (label == 0)).sum().float()  # True Negatives for class 1
    false_positives = ((pred == 1) & (label == 0)).sum().float() # False Positives for class 1

    if (true_negatives + false_positives) > 0:
        rej = true_negatives / (true_negatives + false_positives)
    else:
        rej = torch.tensor(0.0)

    return rej


def acc_binary(pred, label):
    """
    Compute binary classification accuracy.

    Accuracy = Number of correct predictions / Total samples.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted binary labels (0 or 1).
    label : torch.Tensor
        True binary labels (0 or 1).

    Returns
    -------
    torch.Tensor
        Accuracy of predictions.
    """
    correct_preds = (pred == label).sum().float()  # Count of correct predictions
    total_samples = label.size(0)

    if total_samples > 0:
        acc = correct_preds / total_samples
    else:
        acc = torch.tensor(0.0)

    return acc


def eff_n_class(pred, label, n_class=4):
    """
    Compute per-class signal efficiency (recall) for multi-class classification.

    Efficiency for class i = True Positives for class i / Total actual samples of class i.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted logits or probabilities, shape [N, n_class].
    label : torch.Tensor
        True class labels, shape [N].
    n_class : int
        Number of classes.

    Returns
    -------
    torch.Tensor
        Efficiency for each class, shape [n_class].
    """
    pred_argmax = torch.argmax(pred, dim=1)
    eff = torch.zeros(n_class)
    # eff_error = torch.zeros(n_class)  # optional error calculation

    for i in range(n_class):
        true_mask = label == i
        total_true = true_mask.sum()
        if total_true > 0:
            correct_preds = (pred_argmax[true_mask] == label[true_mask]).sum()
            eff[i] = correct_preds.float() / total_true.float()
            # eff_error[i] = compute_efficiency_error(correct_preds.float(), total_true.float())

    return eff  # , eff_error


def rej_n_class(pred, label, n_class=4):
    """
    Compute per-class background rejection for multi-class classification.

    Rejection for class i = TN[i] / (TN[i] + FP[i]), where:
      - TN[i]: True negatives (samples not class i, predicted not class i)
      - FP[i]: False positives (samples not class i, predicted as class i)

    Parameters
    ----------
    pred : torch.Tensor
        Predicted logits or probabilities, shape [N, n_class].
    label : torch.Tensor
        True class labels, shape [N].
    n_class : int
        Number of classes.

    Returns
    -------
    torch.Tensor
        Background rejection for each class, shape [n_class].
    """
    pred_argmax = torch.argmax(pred, dim=1)
    rej = torch.zeros(n_class)
    # rej_err = torch.zeros(n_class)  # optional error calculation

    for i in range(n_class):
        bkg_mask = label != i  # Background for class i
        if bkg_mask.sum() > 0:
            fp = (bkg_mask & (pred_argmax == i)).sum()  # False Positives
            tn = (bkg_mask & (pred_argmax != i)).sum()  # True Negatives
            rej[i] = tn.float() / (tn.float() + fp.float())
            # rej_err[i] = compute_efficiency_error(tn.float(), tn.float() + fp.float())

    return rej  # , rej_err


def acc_n_class(pred, label, n_class=4):
    """
    Compute per-class accuracy for multi-class classification.

    Accuracy for class i = Correct predictions for class i / Total actual samples of class i.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted logits or probabilities, shape [N, n_class].
    label : torch.Tensor
        True class labels, shape [N].
    n_class : int
        Number of classes.

    Returns
    -------
    torch.Tensor
        Accuracy for each class, shape [n_class].
    """
    correct_class = {i: 0 for i in range(n_class)}
    pred_argmax = torch.argmax(pred, dim=1)
    # acc_err = torch.zeros(n_class)  # optional error calculation

    pred_class = {i: (pred_argmax == i).sum() for i in range(n_class)}
    true_class = {i: (label == i).sum() for i in range(n_class)}

    if len(pred) != len(label):
        print("Warning: prediction and label length mismatch in acc_n_class")
        print(len(pred), len(label))
    else:
        for i in range(n_class):
            correct_class[i] = torch.sum(pred_argmax[label == i] == label[label == i])
            # acc_err[i] = compute_efficiency_error(correct_class[i], true_class[i])

    correct_preds = torch.Tensor([correct_class[i] for i in range(n_class)])
    all_label = torch.Tensor(tuple(true_class[i] for i in range(n_class)))

    acc = torch.div(correct_preds, all_label)
    return acc  # , acc_err


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
    #     print("pred", pred)
    correct = 0
    correct_class1 = 0
    correct_class2 = 0
    correct_class3 = 0
    correct_class4 = 0

    pred_argmax = torch.argmax(pred, dim=1)
    #     print("pred_argmax ", pred_argmax)
    #     label_argmax = torch.argmax(data.y,dim=1)

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

def weight_binary_class(dataset,hetero=True):
     """
    Computes inverse-frequency class weights for a 2-class classification task.

    Parameters
    ----------
    dataset : iterable
        A dataset of graph objects with 2-class labels.
    hetero : bool, optional
        If True, assumes heterogeneous graph format

    Returns
    -------
    torch.Tensor
        A tensor of shape [2] containing the class weights.
    """
    num_sample = 0
    true_class = {0: 0, 1: 0}  # Pour deux classes

    for tdata in dataset:
        if hetero:
            y = tdata[('chargedtree', 'to', 'neutrals')].y
        else:
            y = tdata.y
        
        # Compter les instances de chaque classe
        true_class[0] += (y == 0).sum().item()
        true_class[1] += (y == 1).sum().item()
        num_sample += len(y)
    
    print(f"Amount signal: {true_class[1]}, {true_class[1]/num_sample}")
    print(f"Amount bkg: {true_class[0]}, {true_class[0]/num_sample}")
    print(f"Amount total: {num_sample}")

    # Calcul des poids pour chaque classe
    weight_class = {
        0: num_sample / (2 * true_class[0]) if true_class[0] > 0 else 0,
        1: num_sample / (2 * true_class[1]) if true_class[1] > 0 else 0
    }
    
    weight = torch.tensor([weight_class[0], weight_class[1]], dtype=torch.float32)
    print(f"Weights :{weight}")
    return weight
  

def weight_n_class(dataset,hetero=False,n_class=5):
    """
    Computes inverse-frequency class weights for a n-class classification task.

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
        A tensor of shape [n] containing the class weights.
    """
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

def NOW(fmt="%H:%M:%S"):
    """Return the current time formatted as a string."""
    return datetime.now().strftime(fmt)


def msg(obj, fmt="%H:%M:%S"):
    """Print a message prefixed by the current time."""
    print("[{}] ".format(NOW(fmt)), obj)


def batched_predict_proba(model, X, batch_size=100_000):
    """
    Predict probabilities on large data X in batches to avoid memory issues.

    Parameters:
        model: model object with a predict_proba method
        X: input data array
        batch_size: number of samples per batch

    Returns:
        Numpy array of concatenated predicted probabilities for all samples
    """
    probas = []
    for i in range(0, len(X), batch_size):
        batch = X[i:i+batch_size]
        probas.append(model.predict_proba(batch))
    return np.vstack(probas)


def plt_smooth(ax, x, y, yerr, **kwargs):
    """Plot a smooth step curve with error bands on a matplotlib axis."""
    curves = ax.step(x, y, where='mid', linewidth=.75, **kwargs)
    ax.fill_between(x, y - yerr, y + yerr, facecolor=curves[0].get_color(),
                    alpha=.3, step='mid')


def hist(array, weights=None, *, bins=20, range=None, log=False):
    """
    Compute a histogram with optional weights and error estimation.

    Supports linear or logarithmic binning.

    Parameters:
        array: data array (numpy or torch tensor)
        weights: optional weights for each data point
        bins: number of bins or array of bin edges
        range: tuple specifying the (min, max) range
        log: if True, use logarithmic binning

    Returns:
        bins: array of bin edges
        y: weighted histogram counts
        yerr: statistical errors for each bin
    """
    if np.shape(array)[1:] == (2, ):
        array, weights = array.T

    # Convert torch tensors to numpy arrays on CPU if needed
    if isinstance(array, torch.Tensor):
        array = array.cpu().numpy()
    if isinstance(weights, torch.Tensor):
        weights = weights.cpu().numpy()

    if weights is None:
        weights = np.ones(len(array))
    if isinstance(bins, int):
        lo, hi = (np.min(array), np.max(array)) if range is None else range
        bins = (np.logspace(np.log10(lo), np.log10(hi), bins) if log 
                else np.linspace(lo, hi, bins))
    y, _ = np.histogram(array, bins=bins, weights=weights)
    w2, _ = np.histogram(array, bins=bins, weights=weights**2)
    yerr = np.sqrt(w2)
    yerr[yerr == 0] = np.mean(weights)  # Avoid zero errors
    return bins, y * 1.0, yerr


def centers(bins, *, log=False, xerr=False):
    """
    Calculate bin centers for linear or logarithmic bins.

    Parameters:
        bins: array of bin edges
        log: if True, calculate geometric mean centers (log scale)
        xerr: if True, also return asymmetric errors for each bin center

    Returns:
        x: array of bin centers
        err (optional): tuple of lower and upper errors for each center
    """
    x = (np.sqrt(bins[1:] * bins[:-1]) if log else
         0.5 * (bins[1:] + bins[:-1]))
    if not xerr:
        return x
    err = np.array((x - bins[:-1], bins[1:] - x))
    return x, err


def plt_pull(ax, bins, hist, model, err=None):
    """
    Draw a pull plot on a matplotlib axis showing deviations between data and model.

    Parameters:
        ax: matplotlib axis
        bins: bin edges for the histogram
        hist: observed data counts
        model: expected model counts
        err: errors on observed counts (optional, sqrt(hist) if None)

    The pull is (data - model) / error, with special coloring for large pulls.
    """
    if err is None:
        err = hist ** 0.5
    # Avoid division by zero by using smallest positive error
    pull = (hist - model) / np.where(err > 0, err, np.min(err[err > 0]))
    ax.stairs(np.where(abs(pull) < 3, pull, 0), bins, linewidth=.5,
              fill=True, color=(.65, .65, .65), edgecolor='black')
    ax.stairs(np.where(abs(pull) >= 3, pull, 0), bins, linewidth=.5,
              fill=True, color=(.9, .2, .2), edgecolor='black')
    lim = np.max(np.abs(ax.get_ylim()))
    ax.set_ylim(-lim, lim)
    vals = tuple(i for i in (3, 5, 7, 9, 15, 30, 50, 100, 200) if i < lim)
    ax.set_yticks((-vals[-1], vals[-1]) if vals else
                  tuple(t for t in ax.get_yticks() if t != 0 and abs(t) < lim))
    ax.hlines(tuple((i, -i) for i in vals), bins[0], bins[-1],
              linestyle='--', linewidth=.5, color='gray')
    ax.set_ylabel(r'$\frac{\mathrm{data} - \mathrm{fit}}{\sigma}$',
                  loc='center')


def ks_test(responses):
    """
    Perform Kolmogorov-Smirnov tests comparing training and validation samples.

    Parameters:
        responses: dictionary with keys like 'Signal (train)', 'Signal (val)', etc.

    Returns:
        Formatted string reporting KS test p-values for signal and background.
    """
    _, signal = ks_2samp(responses['Signal (train)'][0],
                         responses['Signal (val)'][0])
    _, bkg = ks_2samp(responses['Bkg (train)'][0], responses['Bkg (val)'][0])
    return (f'Kolmogorov-Smirnov test: signal (bkg) probability: '
            f'{signal:.3f} ({bkg:.3f})')


def select_epoch_indices(n_epochs, n_dropped_epochs, n_samples=7):
    """
    Select a set of epoch indices to sample training progress.

    Always includes:
    - The first epoch (index 1)
    - The last few epochs (up to 5)
    - Several evenly spaced intermediate epochs

    Parameters:
        n_epochs: total number of epochs run
        n_dropped_epochs: epochs dropped/not considered
        n_samples: number of indices to return (minimum 2)

    Returns:
        Sorted list of unique 1-based epoch indices.
    """
    total_epochs = n_epochs + n_dropped_epochs - 1  # Adjust for indexing

    if n_samples < 2:
        raise ValueError("n_samples must be at least 2 (first and last).")

    if total_epochs < n_samples:
        # Return all epochs if not enough to sample
        return list(range(1, total_epochs + 1))

    # Reserve some epochs at the end and some in the middle
    n_last = min(5, n_samples - 2)
    n_remaining = n_samples - n_last - 1

    indices = [1]  # Always include first epoch

    if n_remaining > 0:
        start = 2
        end = total_epochs - n_last
        if end >= start:
            inter_indices = torch.linspace(start, end, steps=n_remaining + 1).tolist()
            inter_indices = [round(x) for x in inter_indices]
            indices += inter_indices

    # Add last epochs
    last_epochs = list(range(total_epochs - n_last + 1, total_epochs + 1))
    indices.extend(last_epochs)

    # Remove duplicates and sort
    return sorted(set(indices))
