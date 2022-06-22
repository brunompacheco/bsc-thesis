import matplotlib.pyplot as plt
import numpy as np

def euler_step(y0, t0, N, h):
    return y0 + h * N(t0,y0)

def rk_step(y0, t0, N, h):
    k1 = N(t0,y0)
    k2 = N(t0 + h/2, y0 + h*k1/2)
    k3 = N(t0 + h/2, y0 + h*k2/2)
    k4 = N(t0 + h, y0 + h*k3)

    return y0 + h*(k1 + 2*k2 + 2*k3 + k4)/6

def solve_ode(y0, t0, N, h, n_steps, step_fun):
    ys = [y0,]
    
    for _ in range(n_steps):
        y1 = step_fun(y0, t0, N, h)
        ys.append(y1)

        y0 = y1
        t0 = t0 + h
    
    return ys


if __name__ == '__main__':
    N = lambda t, y: y * np.sin(t) ** 2

    y0 = 1

    # solution
    t = np.linspace(0,200,1000)
    phi = lambda t: np.exp((t - np.sin(t)*np.cos(t))/2)
    plt.plot(t, phi(t), label='$\phi(t)$')

    # Euler
    h = 1.
    n_steps = 200
    euler_x = np.linspace(0, n_steps*h, n_steps+1)
    euler_y = solve_ode(y0, 0., N, h, n_steps, euler_step)
    plt.plot(euler_x, euler_y, '--', marker='o', markersize=2, label='Euler ($h=1.0$)')
    h = .5
    n_steps = 400
    euler_x = np.linspace(0, n_steps*h, n_steps+1)
    euler_y = solve_ode(y0, 0., N, h, n_steps, euler_step)
    plt.plot(euler_x, euler_y, '--', marker='o', markersize=2, label='Euler ($h=0.5$)')

    # Runge-Kutta
    h = 1.
    n_steps = 200
    rk_x = np.linspace(0, n_steps*h, n_steps+1)
    rk_y = solve_ode(y0, 0., N, h, n_steps, rk_step)
    plt.plot(rk_x, rk_y, '--', marker='o', markersize=2, label='RK4 ($h=1.0$)')

    h = 5.
    n_steps = 40
    rk_x = np.linspace(0, n_steps*h, n_steps+1)
    rk_y = solve_ode(y0, 0., N, h, n_steps, rk_step)
    plt.plot(rk_x, rk_y, '--', marker='o', markersize=2, label='RK4 ($h=5.0$)')

    plt.yscale('log')
    plt.grid()
    plt.legend()

    plt.xlim([0,200])
    plt.ylim([1,1e45])
    plt.savefig('ode_solver_comparison.png')

    plt.xlim([0,20])
    plt.ylim([1,1e5])
    plt.savefig('ode_solver_comparison_zoom.png')
