import os

import torch

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import mplhep as hep

hep.style.use(hep.style.LHCb2)




def plot_loss(df, version):
    # Combined output loss w/ the scaling beta
    trn_loss = np.array(df["train/combined_loss"])
    val_loss = np.array(df["val/combined_loss"])
    # Output loss of each criterion
    trn_LCA_loss = np.array(df["train/LCA_loss"])
    val_LCA_loss = np.array(df["val/LCA_loss"])
    trn_PV_loss = np.array(df["train/tPV_edges_loss"])
    val_PV_loss = np.array(df["val/tPV_edges_loss"])
    trn_edge_loss = np.array(df["train/tt_edges_loss"])
    val_edge_loss = np.array(df["val/tt_edges_loss"])
    trn_track_loss = np.array(df["train/t_nodes_loss"])
    val_track_loss = np.array(df["val/t_nodes_loss"])
    #
    epochs = np.arange(len(trn_loss))

    # Plot dir
    outdir = f"lightning_logs/version_{version}/plots"
    os.makedirs(outdir, exist_ok=True)

    # Plot combined loss
    f, ax = plt.subplots(figsize=(9, 6))
    ax.plot(epochs, trn_loss, color="#4169E1", label="trn loss")
    ax.plot(epochs, val_loss, color="#B22222", label="val loss")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.savefig(f"{outdir}/combined_loss.pdf")
    plt.savefig(f"{outdir}/combined_loss.png")
    plt.close()

    # Individual loss
    f, ax = plt.subplots(figsize=(9, 6))
    ax.plot(epochs, trn_LCA_loss, color="black", label="$L^\mathrm{LCA}_\mathrm{CE}$")
    ax.plot(epochs, val_LCA_loss, color="black", linestyle='dashed')

    ax.plot(epochs, trn_edge_loss, color="blue", label="$L^\mathrm{e_{tr-tr}}_\mathrm{BCE}$")
    ax.plot(epochs, val_edge_loss, color="blue", linestyle='dashed')

    ax.plot(epochs, trn_track_loss, color="red", label="$L^\mathrm{v_{tr}}_\mathrm{BCE}$")
    ax.plot(epochs, val_track_loss, color="red", linestyle='dashed')

    ax.plot(epochs, trn_PV_loss, color="green", label="$L^\mathrm{e_{pv-tr}}_\mathrm{BCE}$")
    ax.plot(epochs, val_PV_loss, color="green", linestyle='dashed')
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.savefig(f"{outdir}/individual_loss.pdf")
    plt.savefig(f"{outdir}/individual_loss.png")
    plt.close()


def plot_LCA_acc(df, version):
    trn_LCA_acc0 = np.array(df["train/LCA0_acc"])
    trn_LCA_acc1 = np.array(df["train/LCA1_acc"])
    trn_LCA_acc2 = np.array(df["train/LCA2_acc"])
    trn_LCA_acc3 = np.array(df["train/LCA3_acc"])

    val_LCA_acc0 = np.array(df["val/LCA0_acc"])
    val_LCA_acc1 = np.array(df["val/LCA1_acc"])
    val_LCA_acc2 = np.array(df["val/LCA2_acc"])
    val_LCA_acc3 = np.array(df["val/LCA3_acc"])

    epochs = np.arange(len(trn_LCA_acc0))

    # Plot dir
    outdir = f"lightning_logs/version_{version}/plots"
    os.makedirs(outdir, exist_ok=True)

    # Plot LCA acc
    f, ax = plt.subplots(figsize=(9, 6))
    ax.plot(epochs, trn_LCA_acc0, color="black", label="LCA=0")
    ax.plot(epochs, val_LCA_acc0, color="black", linestyle='dashed')

    ax.plot(epochs, trn_LCA_acc1, color="blue", label="LCA=1")
    ax.plot(epochs, val_LCA_acc1, color="blue", linestyle='dashed')

    ax.plot(epochs, trn_LCA_acc2, color="red", label="LCA=2")
    ax.plot(epochs, val_LCA_acc2, color="red", linestyle='dashed')

    ax.plot(epochs, trn_LCA_acc3, color="green", label="LCA=3")
    ax.plot(epochs, val_LCA_acc3, color="green", linestyle='dashed')

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy [%]")
    ax.legend()
    plt.savefig(f"{outdir}/LCA_acc.pdf")
    plt.savefig(f"{outdir}/LCA_acc.png")
    plt.close()


def plot_fragmentation(df, nlayers, version):

    # Plot dir
    outdir = f"lightning_logs/version_{version}/plots"
    os.makedirs(outdir, exist_ok=True)

    for i in range(nlayers):  
        true_val = df[f"frag_pos_part_score_{i}"]
        false_val = df[f"frag_neg_part_score_{i}"]

        true_weights = np.ones_like(true_val) / len(true_val)
        fake_weights = np.ones_like(false_val) / len(false_val)

        f, ax = plt.subplots(figsize=(9, 6))
        ax.hist(true_val, bins=100, range=[0, 1], alpha=.7, label="Fragmentation", color='#B22222', weights=true_weights)
        ax.hist(false_val, bins=100, range=[0, 1], alpha=.8, label="Remaining", color='#4169E1', weights=fake_weights)

        ax.set_xlabel("NN weights [a.u.]")
        ax.set_ylabel("Normalized entries [a.u.]")
        ax.legend()
        ax.set_yscale("log")
        plt.savefig(f"{outdir}/frag_output_layer_{i}.pdf")
        plt.savefig(f"{outdir}/frag_output_layer_{i}.png")
        plt.close()


def plot_ft_nodes(df, nlayers, version):
    # Plot dir
    outdir = f"lightning_logs/version_{version}/plots"
    os.makedirs(outdir, exist_ok=True)

    labels = df["ft_y"]
    for i in range(nlayers):  
        ft_score = df[f"ft_score_{i}"]
        pred = torch.argmax(ft_score, dim=1)

        cm = confusion_matrix(labels.numpy(), pred.numpy())

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['bbar', 'not b', 'b'], yticklabels=['bbar', 'not b', 'b'], annot_kws={"size": 10})
        plt.xlabel('Predicted', fontsize=18)
        plt.ylabel('True', fontsize=18)
        plt.tight_layout()
        plt.savefig(f"{outdir}/ft_conf_layer_{i}.pdf")
        plt.savefig(f"{outdir}/ft_conf_layer_{i}.png")
        plt.close()
