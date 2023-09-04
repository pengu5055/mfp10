"""
This is an example use of the difference method partial differential equation solver.
"""
import numpy as np
import matplotlib.pyplot as plt
from src import FDMSolver

# Set some constants as far as this case goes
N = 500
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
periods = 11
T = periods * 2*np.pi/omega
M = periods * 1000
t_points = np.linspace(0, T, M)

solver = FDMSolver(initial_condition, harmonic_potential, x_range, t_points, N)
# solution = solver.solve()
a = solver.solve_Analytic(solver.x, solver.t)

# Plot the solution
# solver.plot_Animation(saveVideo=False, filename="case1b.mp4", fps=120)
solver.plot_Heatmap(analytic=True)

# NOTE REMOVE FOR FURTHER PLOTS
quit()

# Instantiate another solver with a different N
solver2 = FDMSolver(initial_condition, harmonic_potential, x_range, t_points, 700)
solution2 = solver2.solve()
solver2.plot_Animation(saveVideo=True, filename="case1b.mp4", fps=500)

# Plot the difference between the two solutions
fig, ax = plt.subplots(facecolor="#4d4c4c")
plt.rcParams.update({'font.family': 'Verdana'})
plt.plot(solver.t,  np.mean(np.abs(solution2)**2, axis=1) - np.mean(np.abs(solution)**2, axis=1), color="#fa84b3")
plt.xlabel("t")
plt.ylabel(r"$|\psi_{N=700}|^2 - |\psi_{N=200}|^2$")
plt.title("Difference between the two solutions - Case 1", color="#dedede")
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
analytic1 = np.abs(solver.solve_Analytic(np.mean(solver.x), solver.t))**2
analytic2 = np.abs(solver2.solve_Analytic(np.mean(solver2.x), solver2.t))**2
print(f"Shape of analytic1: {analytic1.shape}")
print(f"Shape of analytic2: {analytic2.shape}")
avg_sol1 = np.mean(np.abs(solution)**2, axis=1)
avg_sol2 = np.mean(np.abs(solution2)**2, axis=1)
print(f"Shape of avg_sol: {avg_sol1.shape}")
print(f"Shape of avg_sol2: {avg_sol2.shape}")

fig, ax = plt.subplots(facecolor="#4d4c4c")
plt.rcParams.update({'font.family': 'Verdana'})
plt.plot(solver.t, avg_sol1 - analytic1, color="#fa84b3")
plt.plot(solver2.t, avg_sol2 - analytic2, color="#daf589")
plt.xlabel("t")
plt.ylabel(r"$|\psi_{\mathrm{num}}|^2 - |\psi_{\mathrm{ana}}|^2$")
plt.title("Difference between numeric and analytic solution - Case 1", color="#dedede")
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
