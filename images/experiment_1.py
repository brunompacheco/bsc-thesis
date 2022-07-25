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

    df_pinn, pinn_histories = get_runs_data(
        api.runs("brunompac/pideq-vdp", {'group': 'PINN-baseline'}),
        keys=keys
    )
    df_pideq, pideq_histories = get_runs_data(
        api.runs("brunompac/pideq-vdp", {'group': 'PIDEQ-baseline'}),
        keys=keys
    )

    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(6,4)
    fig.patch.set_alpha(.0)

    plot_learning_curve(ax, pinn_histories, 'PINN')
    plot_learning_curve(ax, pideq_histories, 'PIDEQ')

    # ax.set_title('Performance of baseline models')
    ax.set_ylabel('IAE')
    ax.set_xlabel('Epoch')
    ax.set_xlim([0,5e4])
    # ax.set_ylim([0,0.5])
    # ax.set_ylim([1e-4,1e-1])
    ax.set_yscale('log')

    ax.legend()
    ax.grid()

    plt.savefig('exp_1_iae.pdf', bbox_inches='tight')
    # plt.show()

    print("Average training pass time (per epoch):")
    print(f"\tPINN = {pinn_histories['train_time'].mean()*1e3:.3f} ms")
    print(f"\tPIDEQ = {pideq_histories['train_time'].mean()*1e3:.3f} ms")
    print("Average validation pass time (per epoch):")
    print(f"\tPINN = {pinn_histories['val_time'].mean()*1e3:.3f} ms")
    print(f"\tPIDEQ = {pideq_histories['val_time'].mean()*1e3:.3f} ms")

    ### VdP PLOT ###
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    T = 2

    # K = 1000
    # dt = T / K
    # time = [dt * k for k in range(K+1)]

    # y0 = torch.Tensor([0., .1]).unsqueeze(0)

    # u = torch.Tensor([0.]).unsqueeze(0)

    # # y = odeint(lambda t, y: four_tanks(y,u), y0, torch.Tensor(time), method='rk4')
    # y = odeint(lambda t, y: f(y,u), y0, torch.Tensor([i * dt for i in range(K+1)]), method='rk4')
    # y = y.squeeze(1).detach().numpy()

    # t = torch.Tensor(time).unsqueeze(-1).to(device)

    net = load_from_wandb(PIDEQ(T, n_out=2, n_states=80), 'zj7r8add', model_fname='model_last').to(device)
    net.eval()
    pideq_B = net.B.weight.cpu().detach().numpy()
    # pideq_n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # y_pred_pideq = net(t).cpu().detach().numpy()

    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(3,3)
    fig.patch.set_alpha(.0)

    ax.matshow(np.abs(pideq_B), cmap='Blues', vmin=0)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig('exp_1_matplot.pdf', bbox_inches='tight')
    # plt.show()

    # fig, ax = plt.subplots(1,1)
    # fig.set_size_inches(3,3)

    # ax.hist(np.abs(pideq_B).sum(axis=1), bins=20)

    # plt.savefig('exp_1_hist.pdf', bbox_inches='tight')
    # # plt.show()

    # net = load_from_wandb(PINN(T, n_out=2, n_hidden=4, n_nodes=20), 'm5fa0h4m', model_fname='model_last').to(device)
    # net.eval()
    # pinn_n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # y_pred_pinn = net(t).cpu().detach().numpy()

    # print("Number of parameters:")
    # print(f"\tPINN = {pinn_n_params}")
    # print(f"\tPIDEQ = {pideq_n_params}")

    # fig, ax = plt.subplots(1,1)
    # fig.set_size_inches(3,3)

    # ax.plot(time, y[:,0], label='RK4')
    # ax.plot(time, y_pred_pinn[:,0], ':', label='PINN')
    # ax.plot(time, y_pred_pideq[:,0], '--', label='PIDEQ')
    # ax.set_title('$y_1$')
    # ax.set_xlabel('Time [s]')
    # ax.set_xlim([0,2])
    # ax.legend()
    # ax.grid()

    # plt.show()

    # fig, ax = plt.subplots(1,1)
    # fig.set_size_inches(3,3)

    # ax.plot(time, y[:,1], label='RK4')
    # ax.plot(time, y_pred_pinn[:,1], ':', label='PINN')
    # ax.plot(time, y_pred_pideq[:,1], '--', label='PIDEQ')
    # ax.set_title('$y_2$')
    # ax.set_xlabel('Time [s]')
    # ax.set_xlim([0,2])
    # ax.legend()
    # ax.grid()

    # plt.show()
    pass
