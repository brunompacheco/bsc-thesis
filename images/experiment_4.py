from tkinter import W
import wandb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import torch

from torchdiffeq import odeint

from pideq.trainer import f
from pideq.net import PINN, PIDEQ
from pideq.utils import load_from_wandb

from experiment_1 import get_runs_data

plt.rcParams.update({'font.size': 10})
plt.style.use('tableau-colorblind10')


if __name__ == '__main__':
    ### IAE PLOT ###
    api = wandb.Api()

    keys = ['val_loss_iae', 'train_time', 'val_time']

    histories = dict()

    runs = api.runs("brunompac/pideq-vdp", {'$and': [{'group': f'PIDEQ-#z=5'}, {'config.T': 2}, {'config.n_states': 5}]})
    _, hs = get_runs_data(
        runs,
        keys=keys
    )
    histories[1.0] = hs
    for j_l in [0, 2, 0.1]:
        runs = api.runs("brunompac/pideq-vdp", {'$and': [{'group': f'PIDEQ-#jac_lamb={j_l}'}, {'config.T': 2}, {'config.n_states': 5}]})
        _, hs = get_runs_data(
            runs,
            keys=keys
        )
        histories[j_l] = hs

    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(6,4)

    for j_l, hs in histories.items():
        iae_low = hs.groupby('epoch')['val_loss_iae'].min()
        iae_high = hs.groupby('epoch')['val_loss_iae'].max()
        iae = hs.groupby('epoch')['val_loss_iae'].mean()

        ax.fill_between(iae_low.index, iae_low, iae_high, alpha=.5, linewidth=0)
        ax.plot(iae, label=f"$\kappa={j_l:1.1f}$")

    # ax.set_title('Performance of baseline models')
    ax.set_ylabel('IAE')
    ax.set_xlabel('Epoch')
    ax.set_xlim([0,5e4])
    # ax.set_ylim([0,0.5])
    # ax.set_ylim([1e-4,1e-1])
    ax.set_yscale('log')

    ax.legend()
    ax.grid()

    plt.savefig('exp_4_iae.pdf', bbox_inches='tight')
    # plt.show()

    print("Median training pass time (per epoch):")
    for n_z, hs in histories.items():
        print(f"\t{n_z} = {hs['train_time'].median()*1e3:.3f} ms")
    print("Median validation pass time (per epoch):")
    for n_z, hs in histories.items():
        print(f"\t{n_z} = {hs['val_time'].median()*1e3:.3f} ms")
