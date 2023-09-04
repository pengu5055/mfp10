"""
This is an example use of the difference method partial differential equation solver.
This the second case of the project where we study a gaussian wavepacket.
"""
import numpy as np
import matplotlib.pyplot as plt
from src2 import FDMSolver

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

def no_potential(x):
    """
    There is no potential in this case.
    """
    return 0

def analytic_solution(x, t, sigma_0=sigma_0, k_0 = k_0, lamb=lamb):
    """
    The analytic solution for the wavefunction.
    """
    t = np.array(t)
    return (2*np.pi*sigma_0**2)**(-1/4) / np.sqrt(1 + 1j*t/(2*sigma_0**2)) * \
            np.exp((-(x-lamb)**2 / (4*sigma_0**2) + 1j*k_0*(x-lamb) - 1j * k_0**2 * t/2) /(1 + 1j*t/(2*sigma_0**2)))

# Only time is missing
T = 300
M = 11000
t_points = np.linspace(0, T, M)

solver = FDMSolver(initial_condition, no_potential, x_range, t_points, N)
solution1 = solver.solve()

# Plot the solution
solver.plot_Animation(saveVideo=False, filename="case2b.mp4", fps=500)
solver.plot_Heatmap()

# Instantiate another solver with a different N
solver2 = FDMSolver(initial_condition, no_potential, x_range, t_points, 700)
solution2 = solver2.solve()

# Plot the difference between the two solutions
fig, ax = plt.subplots(facecolor="#4d4c4c")
plt.rcParams.update({'font.family': 'Verdana'})
plt.plot(solver.t, np.mean(np.abs(solution1)**2, axis=1) - np.mean(np.abs(solution2)**2, axis=1) , color="#fa84b3")
plt.xlabel("t")
plt.ylabel(r"$|\psi_{N=700}|^2 - |\psi_{N=200}|^2$")
plt.title("Difference between the two solutions - Case 2", color="#dedede")
ax.spines['bottom'].set_color("#dedede")
ax.spines['top'].set_color("#dedede")
ax.spines['right'].set_color("#dedede")
ax.spines['left'].set_color("#dedede")
ax.xaxis.label.set_color("#dedede")
ax.yaxis.label.set_color("#dedede")
ax.tick_params(axis="x", colors="#dedede")
ax.tick_params(axis="y", colors="#dedede")
plt.grid(c="#d1d1d1", alpha=0.5)
plt.show()

# Plot the differemce between the two solutions and the analytic solution
analytic1 = np.abs(solver.analytic_solution(np.mean(solver.x), solver.t))**2
analytic2 = np.abs(solver2.analytic_solution(np.mean(solver2.x), solver2.t))**2
print(f"Shape of analytic1: {analytic1.shape}")
print(f"Shape of analytic2: {analytic2.shape}")
avg_sol1 = np.mean(np.abs(solution1)**2, axis=1)
avg_sol2 = np.mean(np.abs(solution2)**2, axis=1)
print(f"Shape of avg_sol: {avg_sol1.shape}")
print(f"Shape of avg_sol2: {avg_sol2.shape}")

fig, ax = plt.subplots(facecolor="#4d4c4c")
plt.rcParams.update({'font.family': 'Verdana'})
# plt.plot(solver.t, avg_sol1 - analytic1, color="#fa84b3")
plt.plot(solver2.t, avg_sol2 - analytic2, color="#daf589")
plt.xlabel("t")
plt.ylabel(r"$|\psi_{\mathrm{num}}|^2 - |\psi_{\mathrm{ana}}|^2$")
plt.title("Difference between numeric and analytic solution - Case 2", color="#dedede")
ax.spines['bottom'].set_color("#dedede")
ax.spines['top'].set_color("#dedede")
ax.spines['right'].set_color("#dedede")
ax.spines['left'].set_color("#dedede")
ax.xaxis.label.set_color("#dedede")
ax.yaxis.label.set_color("#dedede")
ax.tick_params(axis="x", colors="#dedede")
ax.tick_params(axis="y", colors="#dedede")
plt.grid(c="#d1d1d1", alpha=0.5)
plt.show()
