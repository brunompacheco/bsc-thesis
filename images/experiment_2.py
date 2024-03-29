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

    _, hs = get_runs_data(
        api.runs("brunompac/pideq-vdp", {'$and': [{'group': f'PIDEQ-baseline'}, {'config.T': 2}]}),
        keys=keys
    )
    histories[80] = hs
    for n_z in [40, 20, 10, 5, 2]:
        _df, hs = get_runs_data(
            api.runs("brunompac/pideq-vdp", {'$and': [{'group': f'PIDEQ-#z={n_z}'}, {'config.T': 2}]}),
            keys=keys
        )
        histories[n_z] = hs
        if n_z == 5:
            df = _df

    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(6,4)
    fig.patch.set_alpha(.0)

    for n_z, hs in histories.items():
        plot_learning_curve(ax, hs, n_z, shadow=False)

    # ax.set_title('Performance of baseline models')
    ax.set_ylabel('IAE')
    ax.set_xlabel('Epoch')
    ax.set_xlim([0,5e4])
    # ax.set_ylim([0,0.5])
    # ax.set_ylim([1e-4,1e-1])
    ax.set_yscale('log')

    ax.legend()
    ax.grid()

    # plt.savefig('exp_2_iae.pdf', bbox_inches='tight')
    plt.show()

    print("Median training pass time (per epoch):")
    for n_z, hs in histories.items():
        print(f"\t{n_z} = {hs['train_time'].median()*1e3:.3f} ms")
    print("Median validation pass time (per epoch):")
    for n_z, hs in histories.items():
        print(f"\t{n_z} = {hs['val_time'].median()*1e3:.3f} ms")

    T = 2

    median_pideq = df.sort_values(by='val_loss_iae', ascending=True).iloc[df.shape[0] // 2]['id']

    net = load_from_wandb(PIDEQ(T, n_out=2, n_states=5), median_pideq, model_fname='model_last')
    net.eval()
    pideq_B = net.B.weight.cpu().detach().numpy()
    pideq_n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(3,3)
    fig.patch.set_alpha(.0)

    cbar = ax.matshow(np.abs(pideq_B), cmap='Blues', vmin=0)
    ax.set_xticks([])
    ax.set_yticks([])

    import matplotlib.ticker as ticker
    # plt.colorbar(cbar, format='%.1e')
    plt.colorbar(cbar, format=ticker.FuncFormatter(
        lambda x, pos: np.format_float_scientific(x, precision=1, min_digits=1, exp_digits=1)
    ))
    # plt.savefig('exp_2_matplot.pdf', bbox_inches='tight')
    plt.show()
