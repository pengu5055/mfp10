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

    def solve(self) -> np.ndarray:
        """
        Solve the PDE.
        """
        # Initialize the solution array
        self.solution = np.zeros((len(self.t), self.N))

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