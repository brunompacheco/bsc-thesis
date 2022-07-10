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
    times = {  # PIDEQ, PINN
        'forward': np.array([3.981, 0.4937]),
        'cost': np.array([1.596, 3.796]),
        'backward': np.array([2.757, 2.63]),
        'validation': np.array([0.5121, 0.4796])
    }

    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(3,4)

    prev = np.zeros(2)
    for t in times.keys():
        ax.bar(['PIDEQ', 'PINN'], times[t], 0.5, label=t, bottom=prev)
        prev += times[t]

    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy() 
        ax.text(x+width/2, 
                y+height/2, 
                '{:.2f} ms'.format(height), 
                horizontalalignment='center', 
                verticalalignment='center')

    ax.set_ylim([0,12])
    ax.legend()

    plt.savefig('final_times.pdf', bbox_inches='tight')
    # plt.show()

    # ### IAE PLOT ###
    # api = wandb.Api()

    # keys = ['val_loss_iae', 'train_time', 'val_time']

    # df_pinn_baseline, pinn_baseline_histories = get_runs_data(
    #     api.runs("brunompac/pideq-vdp", {'$and': [{'group': 'PINN-baseline'}, {'config.T': 2}]}),
    #     keys=keys,
    # )
    # df_pideq_baseline, pideq_baseline_histories = get_runs_data(
    #     api.runs("brunompac/pideq-vdp", {'$and': [{'group': 'PIDEQ-baseline'}, {'config.T': 2}]}),
    #     keys=keys,
    # )
    # df_pinn_final, pinn_final_histories = get_runs_data(
    #     api.runs("brunompac/pideq-vdp", {'$and': [{'group': 'PINN-baseline-small'}, {'config.T': 2}]}),
    #     keys=keys,
    # )
    # df_pideq_final, pideq_final_histories = get_runs_data(
    #     api.runs("brunompac/pideq-vdp", {'$and': [{'group': 'PIDEQ-#solver=forward_iteration'}, {'config.T': 2}, {'config.n_states': 5}]}),
    #     keys=keys,
    # )

    # fig, ax = plt.subplots(1,1)
    # fig.set_size_inches(6,4)

    # def plot_learning_curve(ax, histories, label):
    #     iae_low = histories.groupby('epoch')['val_loss_iae'].min()
    #     iae_high = histories.groupby('epoch')['val_loss_iae'].max()
    #     iae = histories.groupby('epoch')['val_loss_iae'].mean()

    #     ax.fill_between(iae_low.index, iae_low, iae_high, alpha=.5, linewidth=0)
    #     ax.plot(iae, label=label)

    # plot_learning_curve(ax, pinn_baseline_histories, 'Baseline PINN')
    # plot_learning_curve(ax, pideq_baseline_histories, 'Baseline PIDEQ')
    # plot_learning_curve(ax, pideq_final_histories, 'Final PIDEQ')
    # plot_learning_curve(ax, pinn_final_histories, 'Final PINN')

    # # ax.set_title('Performance of baseline models')
    # ax.set_ylabel('IAE')
    # ax.set_xlabel('Epoch')
    # ax.set_xlim([0,5e4])
    # # ax.set_ylim([0,0.5])
    # # ax.set_ylim([1e-4,1e-1])
    # ax.set_yscale('log')

    # ax.legend()
    # ax.grid()

    # plt.savefig('final_iae.pdf', bbox_inches='tight')
    # # plt.show()

    ### VdP PLOT ###
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # median_pinn = df_pinn_final.sort_values(by='val_loss_iae', ascending=True).iloc[df_pinn.shape[0] // 2]['id']
    # median_pideq = df_pideq_final.sort_values(by='val_loss_iae', ascending=True).iloc[df_pideq.shape[0] // 2]['id']

    # T = 2

    # K = 1000
    # dt = T / K
    # time = [dt * k for k in range(K+1)]

    # y0 = torch.Tensor([0., .1]).unsqueeze(0)

    # u = torch.Tensor([0.]).unsqueeze(0)

    # # y = odeint(lambda t, y: four_tanks(y,u), y0, torch.Tensor(time), method='rk4')
    # y = odeint(lambda t, y: f(y,u), y0, torch.Tensor([i * dt for i in range(K+1)]), method='rk4')
    # y = y.squeeze(1).detach().numpy()

    # t = torch.Tensor(time).unsqueeze(-1).to(device)

    # net = load_from_wandb(PIDEQ(T, n_out=2, n_states=5), median_pideq, model_fname='model_last').to(device)
    # net.eval()
    # pideq_n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # y_pred_pideq = net(t).cpu().detach().numpy()

    # net = load_from_wandb(PINN(T, n_out=2, n_hidden=2, n_nodes=5), median_pinn, model_fname='model_last').to(device)
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

    # plt.savefig("final_vdp_y1.pdf", bbox_inches='tight')
    # # plt.show()

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

    # plt.savefig("final_vdp_y2.pdf", bbox_inches='tight')
    # # plt.show()
