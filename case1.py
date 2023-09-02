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
t_points = np.linspace(0, 10, 1000)

solver = FDMSolver(initial_condition, harmonic_potential, x_range, t_points, N)
solution = solver.solve()

# Plot the solution
fig, ax = plt.subplots()
ax.plot(solver.x, np.abs(solution[0])**2, label="Numerical")
ax.plot(solver.x, np.abs(analytic_solution(solver.x, 0, lamb=lamb, k=omega**2))**2, label="Analytic")
ax.set_xlabel("x")
ax.set_ylabel(r"$|\psi(x,t)|^2$")
ax.set_title("t = 0")
ax.legend()
plt.show()
