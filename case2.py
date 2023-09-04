"""
This is an example use of the difference method partial differential equation solver.
This the second case of the project where we study a gaussian wavepacket.
"""
import numpy as np
import matplotlib.pyplot as plt
from src import FDMSolver

# Set some constants as far as this case goes
N = 200
x_range = (-0.5, 1.5)
sigma_0 = 1/20
k_0 = 50*np.pi
lamb = 0.25


def initial_condition(x: np.ndarray, sigma_0=sigma_0, k_0=k_0, lamb=lamb) -> np.ndarray:
    """
    The initial condition for the wavefunction.
    """
    return (2*np.pi*sigma_0**2)**(-1/4) * np.exp(1j*k_0*(x-lamb) - (x-lamb)**2/(4*sigma_0**2))

def no_potential(x, k = k):
    """
    There is no potential in this case.
    """
    return 0

def analytic_solution(x, t, sigma_0=sigma_0, k_0 = k_0, lamb=lamb):
    """
    The analytic solution for the wavefunction.
    """
    return (2*np.pi*sigma_0**2)**(-1/4) / np.sqrt(1 + 1j*t/(2*sigma_0**2)) * \
            np.exp((-(x-lamb)**2 / (4*sigma_0**2) + 1j*k_0*(x-lamb) - 1j * k_0**2 * t/2) /(1 + 1j*t/(2*sigma_0**2)))


# Only time is missing
# We need to observe 10 periods T = 2pi/omega
periods = 11
T = periods * 10
M = periods * 1000
t_points = np.linspace(0, T, M)

solver = FDMSolver(initial_condition, no_potential, x_range, t_points, N)
solution = solver.solve()

# Plot the solution
solver.plot_Animation(saveVideo=False, filename="case2b.mp4", fps=120)
solver.plot_Heatmap()
