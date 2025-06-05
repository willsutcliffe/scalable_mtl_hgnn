import torch
import numpy as np
from torch_scatter import scatter_add
from datetime import datetime
from scipy.stats import ks_2samp



def neutrals_hetero_positive_edge_weight(loader):
    sum_edges = 0
    sum_pos = 0
    for data in loader:
        #print("A:\t",data[('tracks','to','tracks')].edges.shape[0])
        #print("B:\t",torch.sum(data[('tracks','to','tracks')].y[:,0]==0).item())
        sum_edges += data[('chargedtree','to','neutrals')].edges.shape[0]
        sum_pos  += torch.sum(data[('chargedtree','to','neutrals')].y[:,0]==0).item()
    return sum_edges/(2*sum_pos)

### TODO modify functions to adapt neutrals
def neutrals_hetero_positive_node_weight(loader):
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
    sum_edges = 0
    sum_pos = 0
    for data in loader:
        #print("A:\t",data[('tracks','to','tracks')].edges.shape[0])
        #print("B:\t",torch.sum(data[('tracks','to','tracks')].y[:,0]==0).item())
        sum_edges += data[('tracks','to','tracks')].edges.shape[0]
        sum_pos  += torch.sum(data[('tracks','to','tracks')].y[:,0]==0).item()
    return sum_edges/(2*sum_pos)

def hetero_positive_node_weight(loader):
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
    sum_edges = 0
    sum_pos = 0
    for data in loader:
        print("D:\t",data.edges.shape[0])
        print("E:\t",torch.sum(data.y[:,0]==0).item())
        sum_edges += data.edges.shape[0]
        sum_pos  += torch.sum(data.y[:,0]==0).item()
    return sum_edges/(2*sum_pos)

def positive_node_weight(loader):
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

def compute_efficiency_error(num,den):
    """
    eff = num / den
    but den = num + a
    """
    a = den - num
    return torch.sqrt(num*a/((num+a)**3))

def eff_binary(pred, label):
    """
    Compute binary signal efficiency (recall) for class 1:
    eff = true positives for class 1 / total actual samples of class 1
    """
    
    true_positives = (pred * label).sum().float()  # TP for class 1
    total_positives = label.sum().float()  # Total actual samples of class 1
    

    if total_positives > 0:
        eff = true_positives / total_positives
    else:
        eff = torch.tensor(0.0)

    return eff

def rej_binary(pred, label):
    """
    Compute binary background rejection for class 1:
    rej = TN[1] / (TN[1] + FP[1])
    """
    true_negatives = ((pred == 0) & (label == 0)).sum().float()  # TN for class 1
    false_positives = ((pred == 1) & (label == 0)).sum().float()  # FP for class 1
    
    if (true_negatives + false_positives) > 0:
        rej = true_negatives / (true_negatives + false_positives)
    else:
        rej = torch.tensor(0.0)
    return rej

def acc_binary(pred, label):
    """
    Compute binary accuracy:
    acc = correct predictions / total samples
    """
    correct_preds = (pred == label).sum().float()  # Correct predictions
    total_samples = label.size(0)
    
    if total_samples > 0:
        acc = correct_preds / total_samples
    else:
        acc = torch.tensor(0.0)
    return acc

def eff_n_class(pred, label, n_class=4):
    """
    Compute per-class signal efficiency (recall):
    eff[i] = true positives for class i / total actual samples of class i
    """
    pred_argmax = torch.argmax(pred, dim=1)
    eff = torch.zeros(n_class)
    eff_error = torch.zeros(n_class)
    
    for i in range(n_class):
        true_mask = label == i
        total_true = true_mask.sum()
        if total_true > 0:
            correct_preds = (pred_argmax[true_mask] == label[true_mask]).sum()
            eff[i] = correct_preds.float() / total_true.float()
            #eff_error[i] = compute_efficiency_error(correct_preds.float(),total_true.float())


    return eff#,eff_error

def rej_n_class(pred, label, n_class=4):
    """
    Compute per-class background rejection:
    rej[i] = TN[i] / (TN[i] + FP[i])
    Where:
        - TN[i]: true label != i and predicted label != i
        - FP[i]: true label != i and predicted label == i
    """
    pred_argmax = torch.argmax(pred, dim=1)
    rej = torch.zeros(n_class)
    rej_err = torch.zeros(n_class)

    for i in range(n_class):
        bkg_mask = label != i  # background for class i
        if bkg_mask.sum() > 0:
            fp = (bkg_mask & (pred_argmax == i)).sum()   # predicted as i but shouldn't be
            tn = (bkg_mask & (pred_argmax != i)).sum()   # correctly not predicted as i
            rej[i] = tn.float() / (tn.float() + fp.float())
            #rej_err[i] = compute_efficiency_error(tn.float(), tn.float() + fp.float())

    return rej#, rej_err


def acc_n_class(pred, label, n_class=4):
    correct = 0
    correct_class = {i : 0 for i in range(n_class)}
    pred_argmax = torch.argmax(pred, dim=1)
    acc_err = torch.zeros(n_class)
    
    pred_class = {i : (pred_argmax == i).sum() for i in range(n_class)}
    true_class = {i : (label == i).sum() for i in range(n_class)}
    
    if len(pred) != len(label):
        print("something goes wrong in acc_n_class")
        print(len(pred), len(label))
    else:
        for i in range(n_class):
            correct_class[i] = torch.sum(pred_argmax[label == i] == label[label == i])
            #acc_err[i] = compute_efficiency_error(correct_class[i], true_class[i])

    correct_preds = torch.Tensor([correct_class[i] for i in range(n_class)])
    all_preds = torch.Tensor(tuple(pred_class[i] for i in range(n_class)))
    all_label = torch.Tensor(tuple(true_class[i] for i in range(n_class)))

    acc = torch.div(correct_preds, all_label)
    return acc#, acc_err
    

def acc_four_class(pred, label):
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
    
    # Calcul des poids pour chaque classe
    weight_class = {
        0: num_sample / (2 * true_class[0]) if true_class[0] > 0 else 0,
        1: num_sample / (2 * true_class[1]) if true_class[1] > 0 else 0
    }
    
    weight = torch.tensor([weight_class[0], weight_class[1]], dtype=torch.float32)
    print(f"Weights :{weight}")
    return weight
  

def weight_n_class(dataset,hetero=False,n_class=5):
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
    """return current time formatted"""
    return datetime.now().strftime(fmt)

def msg(obj,fmt="%H:%M:%S"):
    """print string with time information"""
    print("[{}] ".format(NOW(fmt)),obj)

def batched_predict_proba(model, X, batch_size=100_000):
    probas = []
    for i in range(0, len(X), batch_size):
        batch = X[i:i+batch_size]
        probas.append(model.predict_proba(batch))
    return np.vstack(probas)

def plt_smooth(ax, x, y, yerr, **kwargs):
    """Plot a smooth curve with errors"""
    curves = ax.step(x, y, where='mid', linewidth=.75, **kwargs)
    ax.fill_between(x, y - yerr, y + yerr, facecolor=curves[0].get_color(),
                    alpha=.3, step='mid')

def hist(array, weights=None, *, bins=20, range=None, log=False):
    """Create a histogram (with errors)"""
    if np.shape(array)[1:] == (2, ):
        array, weights = array.T

    # Check if array is a torch tensor and move to CPU
    if isinstance(array, torch.Tensor):
        array = array.cpu().numpy()  # Move to CPU and convert to numpy array
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
    yerr = w2**0.5
    yerr[yerr == 0] = np.mean(weights)
    return bins, y * 1.0, yerr

def centers(bins, *, log=False, xerr=False):
    """Get bin centers for linear and logarithmic spaces"""
    x = (np.sqrt(bins[1:] * bins[:-1]) if log else
         .5 * (bins[1:] + bins[:-1]))
    if not xerr: return x
    err = np.array((x - bins[:-1], bins[1:] - x))
    return x, err


def plt_pull(ax, bins, hist, model, err=None):
    """Create a pull plot"""
    if err is None: err = hist**.5
    # This min(err > 0) is a guess. The fully correct way would be 1/integral,
    # but when we have zeros, it's likely we also have ones.
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
    """Perform a Kolmogorov-Smirnov test and summarize it"""
    _, signal = ks_2samp(responses['Signal (train)'][0],
                         responses['Signal (val)'][0])
    _, bkg = ks_2samp(responses['Bkg (train)'][0], responses['Bkg (val)'][0])
    return (f'Kolmogorov-Smirnov test: signal (bkg) probability: '
            f'{signal:.3f} ({bkg:.3f})')


def select_epoch_indices(n_epochs, n_dropped_epochs, n_samples=5):
    """
    Return `n_samples` epoch indices (0-based, going up to n_epochs+1),
    including the first (0) and last (n_epochs+1), and equally spaced values in between.
    """
    # Include the first and last epochs
    total_epochs = n_epochs + n_dropped_epochs -1 

    if n_samples < 2:
        raise ValueError("At least two samples are needed (first and last).")

    if n_samples >= total_epochs + 1:
        # Return all indices from 0 to n_epochs + 1
        return list(range(total_epochs + 1))

    # Create equally spaced indices
    indices = torch.linspace(1, total_epochs, steps=n_samples).tolist()

    # Round and convert to a set of integers to avoid duplicates,
    # then sort the result
    return sorted(set(round(x) for x in indices))