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
                       color: str = "black",
                       saveVideo: bool = False, 
                       videoName: str = "animation.mp4", 
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
            videoName: The name of the video to save.
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
            line.set_ydata(solution[frame])
            line.set_color(color)
            L.get_texts()[0].set_text(f"t = {frame/fps:.2f} s")
            return line,

        fig, ax = plt.subplots()
        line, = ax.plot(x, solution[0], label="t = 0 s", c=color)
        ax.set_xlabel("x")
        ax.set_ylabel("T")
        # ax.set_ylim(-1.5, 1.5)
        # ax.set_xlim(0, 1)
        plt.suptitle("Solution of the heat equation")
        L = plt.legend()

        ani = FuncAnimation(fig, update, frames=range(len(self.t_points)), blit=False, interval=1000/fps)
        plt.rcParams['animation.ffmpeg_path'] ='C:\\Media\\ffmpeg\\bin\\ffmpeg.exe' 
        if saveVideo:
            writervideo = FFMpegWriter(fps=fps)
            ani.save(videoName, writer=writervideo)

        plt.show()

    def plot_Basic(self, t_index):
        """
        Plot the solution at a given time index.
        """
        fig, ax = plt.subplots()
        plt.plot(self.x, np.abs(self.solution[t_index])**2, label=f"Numerical t={t_index*self.dt:.2f}")
        plt.plot(self.x, np.abs(self.analytic_solution(self.x, t_index*self.dt))**2, label=f"Analytic t={t_index*self.dt:.2f}")
        plt.xlabel("x")
        plt.ylabel(r"$|\psi(x,t)|^2$")
        plt.title(f"t = {t_index*self.dt:.2f}")
        plt.legend()
        plt.show()


    def analytic_solution(self, x, t, alpha=0.447213595499958, lamb=10, k=0.04000000000000001):
        """
        The analytic solution for the wavefunction in case 1.
        """
        omega = np.sqrt(k)
        xl = alpha * lamb
        xi = alpha * x

        return np.sqrt(alpha/ np.sqrt(np.pi)) * np.exp(-0.5*(xi - xl*np.cos(omega*t))**2 -
                -1j * (omega*t/2 + xi*xl*np.sin(omega*t) - 0.25 * xl**2 * np.sin(2*omega*t))) 