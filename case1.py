"""
This is an example use of the difference method partial differential equation solver.
"""
import numpy as np
import matplotlib.pyplot as plt
from src import FDMSolver

# Set some constants as far as this case goes
N = 300
x_range = (-40,40)
omega = 0.2
lamb = 10
k = omega**2
alpha = k**0.25


def initial_condition(x: np.ndarray, alpha=alpha, lamb=lamb) -> np.ndarray:
    """
    The initial condition for the wavefunction.
    """
    return np.sqrt(alpha/ np.sqrt(np.pi)) * np.exp((-alpha**2 * (x - lamb)**2) / 2)

def harmonic_potential(x, k = k):
    """
    The harmonic oscillator potential.
    """
    return 0.5 * k * x**2

def analytic_solution(x, t, alpha=alpha, lamb=lamb, k=k):
    """
    The analytic solution for the wavefunction.
    """
    omega = np.sqrt(k)
    xl = alpha * lamb
    xi = alpha * x

    return np.sqrt(alpha/ np.sqrt(np.pi)) * np.exp(-0.5*(xi - xl*np.cos(omega*t))**2 -
            -1j * (omega*t/2 + xi*xl*np.sin(omega*t) - 0.25 * xl**2 * np.sin(2*omega*t))) 


# Only time is missing
# We need to observe 10 periods T = 2pi/omega
T = 2*np.pi/omega
t_points = np.linspace(0, T, 1000)

solver = FDMSolver(initial_condition, harmonic_potential, x_range, t_points, N)
solution = solver.solve()
analytic = analytic_solution(solver.x, 0, lamb=lamb, k=omega**2)

# Plot the solution
solver.plot_Animation(saveVideo=True, filename="case1.mp4", fps=120)