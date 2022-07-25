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

from experiments import get_runs_data, plot_learning_curve

plt.rcParams.update({'font.size': 10})
plt.style.use('tableau-colorblind10')


if __name__ == '__main__':
    ### IAE PLOT ###
    api = wandb.Api()

    keys = ['val_loss_iae', 'train_time', 'val_time']

    histories = dict()

    runs = api.runs("brunompac/pideq-vdp", {'$and': [{'group': f'PIDEQ-#solver=forward_iteration'}, {'config.T': 2}, {'config.n_states': 5}]})
    _, hs = get_runs_data(
        runs,
        keys=keys
    )
    histories['1e-04'] = hs
    for eps in ['1e-02', '1e-06']:
        runs = api.runs("brunompac/pideq-vdp", {'$and': [{'group': f'PIDEQ-#eps={eps}'}, {'config.T': 2}, {'config.n_states': 5}]})
        _, hs = get_runs_data(
            runs,
            keys=keys
        )
        histories[eps] = hs

    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(6,4)
    fig.patch.set_alpha(.0)

    plot_learning_curve(ax, histories['1e-02'], '$\\varepsilon=10^{-2}$', shadow=False)
    plot_learning_curve(ax, histories['1e-04'], '$\\varepsilon=10^{-4}$', shadow=False)
    plot_learning_curve(ax, histories['1e-06'], '$\\varepsilon=10^{-6}$', shadow=False)

    # ax.set_title('Performance of baseline models')
    ax.set_ylabel('IAE')
    ax.set_xlabel('Epoch')
    ax.set_xlim([0,5e4])
    # ax.set_ylim([0,0.5])
    # ax.set_ylim([1e-4,1e-1])
    ax.set_yscale('log')

    ax.legend()
    ax.grid()

    plt.savefig('exp_6_iae.pdf', bbox_inches='tight')
    # plt.show()

    print("Median training pass time (per epoch):")
    for n_z, hs in histories.items():
        print(f"\t{n_z} = {hs['train_time'].median()*1e3:.3f} ms")
    print("Median validation pass time (per epoch):")
    for n_z, hs in histories.items():
        print(f"\t{n_z} = {hs['val_time'].median()*1e3:.3f} ms")
