import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from itertools import product


if __name__ == '__main__':

    def N(t, y, mu):
        dy1 = y[1]
        dy2 = mu * (1 - y[0]**2)*y[1] - y[0]

        return np.array([dy1, dy2])

    y0 = np.array([0., .1])

    figsize = (1.9,1.9)
    plt.rcParams.update({'font.size': 10})

    for mu in [0., .5, 1.]:
        t_span = (0, 50)
        sol = solve_ivp(N, t_span, y0, args=(mu,), max_step=.1)

        # time
        plt.figure(figsize=figsize,dpi=300)
        plt.plot(sol.t, sol.y[0], label='$y_1$')
        plt.plot(sol.t, sol.y[1], label='$y_2$')
        plt.xlim(t_span)
        plt.xlabel('$t$')
        plt.gca().set_title(f'$\mu={mu:.1f}$')
        if mu == 0:
            plt.legend()
        plt.savefig(f'vdp_timeplot_mu_{mu*10:02.0f}.pdf', bbox_inches='tight')

        # state-space
        lim = max(-sol.y.min(),sol.y.max())
        lim = int(2*(lim+.5))/2
        quiver_range = np.linspace(-lim, lim, 10)
        quiver_xy = product(quiver_range,quiver_range)
        quiver_xy = np.array(list(quiver_xy))

        quiver_uv = np.array([N(0,np.array(y),mu) for y in quiver_xy])
        quiver_u = quiver_uv[:,0]
        quiver_v = quiver_uv[:,1]

        plt.figure(figsize=figsize,dpi=300)
        plt.quiver(quiver_xy[:,0], quiver_xy[:,1], quiver_uv[:,0], quiver_uv[:,1], width=2e-3)
        plt.plot(sol.y[0], sol.y[1])
        plt.gca().set_title(f'$\mu={mu:.1f}$')
        plt.xlim((-lim,lim))
        plt.ylim((-lim,lim))
        plt.savefig(f'vdp_statespace_mu_{mu*10:02.0f}.pdf', bbox_inches='tight')
