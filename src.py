"""
This is the source script for the difference method partial differential equation solver.
It will be used to solve Schrodinger's equation in 1D for a particle trapped in a harmonic 
oscillator potential. The method used is the difference method, which is a finite difference
method. The difference method is a numerical method for solving partial differential equations
(PDEs) that uses finite difference approximations to the derivatives of the unknown function
to approximate the PDE. The difference method is used to solve the PDEs numerically, and the
resulting difference equations are solved using standard linear algebra techniques, such as
matrix inversion and Gaussian elimination. The difference method is a very general method that
can be used to solve a wide variety of PDEs, including elliptic, parabolic, and hyperbolic
PDEs.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib as mpl
import cmasher as cmr
from typing import Callable, Tuple, Iterable



class FDMSolver():
    def __init__(self,
                 initial_condition: Callable[[np.ndarray], np.ndarray],
                 potential: Callable[[np.ndarray], np.ndarray],
                 x_range: Tuple[float, float],
                 t_points: Iterable[float],
                 N: int,
                 ) -> None:
        """
        Initialize the solver.
        """
        self.initial_condition = initial_condition
        self.potential = potential
        self.x_range = x_range
        self.t = t_points
        self.N = N
        self.x = np.linspace(x_range[0], x_range[1], N)
        self.dx = self.x[1] - self.x[0]
        self.dt = self.t[1] - self.t[0]

        # Initialize the solution array
        self.solution = np.empty((len(self.t), self.N), dtype=np.complex128)

    def solve(self) -> np.ndarray:
        """
        Solve the PDE.
        """
        # Set the initial condition
        self.solution[0] = self.initial_condition(self.x)
        
        # Set the boundary conditions
        # I think this might be a potential error. It should be 0 across all indices
        self.solution[:, 0] = 0
        self.solution[:, -1] = 0

        # Set the potential
        self.V = self.potential(self.x)

        # Set up the matrix
        self.b = 1j*(self.dt/(2*self.dx**2))
        self.a = - self.b/2
        self.d = 1 + self.b + 1j * self.dt/2 * self.V

        self.A = np.diag(self.a * np.ones(self.N-1), -1) + \
            np.diag(self.d * np.ones(self.N), 0) + \
            np.diag(self.a * np.ones(self.N-1), 1)

        self.A_inv = np.linalg.inv(self.A)
        self.A_dagger = np.conj(self.A).T

        # Solve the PDE
        for i in range(1, len(self.t)):
            self.solution[i] = self.A_inv @ self.A_dagger @ self.solution[i-1]
        
        return self.solution
    
    def plot_Animation(self, x: Iterable[float] | None = None, 
                       solution: Iterable[float] | None = None,
                       color: str = "#f07ab9",
                       color2: str = "#faed8c",
                       saveVideo: bool = False, 
                       filename: str = "animation.mp4", 
                       fps: int = 20,
                    ):
        """
        Plot the solution as an animation. Will try to get computed solution
        from solver itself. Can override if x, solution are not 'None'.

        The solution is plotted as an animation. The animation can be saved.

        Arguments:
            x: The grid points at which the solution is evaluated.
            solution: The solution to the heat equation.
            method: The method used to solve the heat equation. Can be either
                "analytical" or "numerical".
            color: The color of the plotted solution.
            saveVideo: Whether or not to save the animation as a video.
            filename: The name of the video to save.
            fps: The frames per second of the video.
        
        Return:
            None
        """
        if np.all(x == None) and np.all(solution == None):
            try:
                x = self.x
                solution = self.solution
            except NameError:
                print("Call one of solving methods before trying to plot or supply data as function parameters!")

        def update(frame):
            line.set_ydata(np.abs(solution[frame])**2)
            line.set_color(color)
            line2.set_ydata(np.abs(self.solve_Analytic(x, frame*self.dt))**2)
            line2.set_color(color2)
            title.set_text(f"t = {frame*self.dt:.2f} s")
            return line,

        plt.rcParams.update({'font.family': 'Verdana'})
        fig, ax = plt.subplots(facecolor="#4d4c4c")

        line, = ax.plot(x, np.abs(solution[0])**2, label="Numeric sol.", c=color)
        line2, = ax.plot(x, np.abs(self.solve_Analytic(x, 0))**2, label="Analytic sol.", c=color2, ls="--")
        line3, = ax.plot(x, self.V, label="Potential", c="#d4ed82", ls=":")

        ax.set_xlabel("x")
        ax.set_ylabel("T")
        plt.suptitle("Schrodinger's equation - Case 1", color="#dedede")
        title = plt.title(f"t={0:.2f} s", color="#dedede")
        L = plt.legend()
        ax.set_ylim(-0.1, 0.5)
        ax.set_xlim(-15, 15)

        ax.set_facecolor("#bababa")
        plt.grid(c="#d1d1d1", alpha=0.5)
        ax.spines['bottom'].set_color("#dedede")
        ax.spines['top'].set_color("#dedede")
        ax.spines['right'].set_color("#dedede")
        ax.spines['left'].set_color("#dedede")
        ax.xaxis.label.set_color("#dedede")
        ax.yaxis.label.set_color("#dedede")
        ax.tick_params(axis="x", colors="#dedede")
        ax.tick_params(axis="y", colors="#dedede")
        ax.axhline(0, linestyle="--", color="#dedede")

        ani = FuncAnimation(fig, update, frames=range(len(self.t)), blit=False, interval=1000/fps)
        plt.rcParams['animation.ffmpeg_path'] ='C:\\Media\\ffmpeg\\bin\\ffmpeg.exe' 
        if saveVideo:
            writervideo = FFMpegWriter(fps=fps)
            ani.save(filename, writer=writervideo)

        plt.show()

    def plot_Basic(self, t_index):
        """
        Plot the solution at a given time index.
        """
        fig, ax = plt.subplots()
        plt.plot(self.x, np.abs(self.solution[t_index])**2, label=f"Numerical t={t_index*self.dt:.2f}")
        plt.plot(self.x, np.abs(self.solve_Analytic(self.x, t_index*self.dt))**2, label=f"Analytic t={t_index*self.dt:.2f}")
        plt.xlabel("x")
        plt.ylabel(r"$|\psi(x,t)|^2$")
        plt.title(f"t = {t_index*self.dt:.2f}")
        plt.legend()
        plt.show()


    def solve_Analytic(self, x, t, alpha=0.447213595499958, lamb=10, k=0.04000000000000001):
        """
        The analytic solution for the wavefunction in case 1.
        Solving it pointwise and not vectorized because no time 
        to vectorize.
        """
        omega = np.sqrt(k)
        xl = alpha * lamb

        self.analytic = np.empty((len(t), len(x)), dtype=np.complex128)

        for i, x_i in enumerate(x):
            for j, t_j in enumerate(t):
                xi = alpha * x_i
                self.analytic[j, i] = np.sqrt(alpha/ np.sqrt(np.pi)) * np.exp(-0.5*(xi - xl*np.cos(omega*t_j))**2 -
                -1j * (omega*t_j/2 + xi*xl*np.sin(omega*t_j) - 0.25 * xl**2 * np.sin(2*omega*t_j)))
        
        return self.analytic
    
    def plot_Heatmap(self, analytic: bool = False):
        """
        Plot the solution as a heatmap.
        """
        plt.rcParams.update({'font.family': 'Verdana'})
        fig, ax = plt.subplots(facecolor="#4d4c4c")
        try:
            if analytic:
                data = np.abs(self.analytic)**2
            else:
                data = np.abs(self.solution)**2
            data = np.flip(data, axis=0)
        except AttributeError:
            print("Call solve method before trying to plot!")
        
        norm = mpl.colors.Normalize(vmin=np.min(data), vmax=np.max(data))
        # norm = mpl.colors.Normalize(vmin=0, vmax=1)   
        plt.imshow(data, cmap=cmr.ghostlight, aspect="auto", norm=norm,# vmin=np.min(data), vmax=np.max(data),
                   extent=[self.x_range[0], self.x_range[1], self.t[0], self.t[-1]])

        x_ticks = np.linspace(self.x_range[0],self.x_range[1], 10)
        plt.xticks(x_ticks)
        y_ticks = np.linspace(self.t[0], self.t[-1], 10)
        plt.yticks(y_ticks)

        plt.xlabel(r"$x\>[arb. units]$")
        plt.ylabel(r"$t\>[arb. units]$")
        plt.suptitle("Heatmap of the Analytic solution - Harmonic Potential", color="#dedede")
        
        t_step = (self.t[-1] - self.t[0])/len(self.t)

        plt.title(f"M = {len(self.t)}, N = {self.N}, t_step = {t_step:.2e}", color="#dedede")

        scalar_Mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmr.ghostlight)
        cb = plt.colorbar(scalar_Mappable, ax=ax, label=r"|\psi(x,t)|^2$",
                      orientation="vertical")
        cb.set_label(r"$|\psi(x,t)|^2$", color="#dedede")
        cb.ax.xaxis.set_tick_params(color="#dedede")
        cb.ax.yaxis.set_tick_params(color="#dedede")
        cb.ax.tick_params(axis="x", colors="#dedede")
        cb.ax.tick_params(axis="y", colors="#dedede")
        ax.spines['bottom'].set_color("#dedede")
        ax.spines['top'].set_color("#dedede")
        ax.spines['right'].set_color("#dedede")
        ax.spines['left'].set_color("#dedede")
        ax.xaxis.label.set_color("#dedede")
        ax.yaxis.label.set_color("#dedede")
        ax.tick_params(axis="x", colors="#dedede")
        ax.tick_params(axis="y", colors="#dedede")
        plt.subplots_adjust(right=.98)
        plt.show()