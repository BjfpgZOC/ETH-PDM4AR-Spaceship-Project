import sympy as spy

from dg_commons.sim.models.spaceship_structures import SpaceshipGeometry, SpaceshipParameters


class SpaceshipDyn:
    sg: SpaceshipGeometry
    sp: SpaceshipParameters

    x: spy.Matrix
    u: spy.Matrix
    p: spy.Matrix

    n_x: int
    n_u: int
    n_p: int

    f: spy.Function
    A: spy.Function
    B: spy.Function
    F: spy.Function

    def __init__(self, sg: SpaceshipGeometry, sp: SpaceshipParameters):
        self.sg = sg
        self.sp = sp

        self.x = spy.Matrix(spy.symbols("x y psi vx vy dpsi delta m", real=True))  # states
        self.u = spy.Matrix(spy.symbols("thrust ddelta", real=True))  # inputs
        self.p = spy.Matrix([spy.symbols("t_f", positive=True)])  # final time

        self.n_x = self.x.shape[0]  # number of states
        self.n_u = self.u.shape[0]  # number of inputs
        self.n_p = self.p.shape[0]

    def get_dynamics(self) -> tuple[spy.Function, spy.Function, spy.Function, spy.Function]:
        """
        Define dynamics and extract jacobians.
        Get dynamics for SCvx.
        0x 1y 2psi 3vx 4vy 5dpsi 6delta 7m
        """

        l_r = self.sg.l_r
        I_z = self.sg.Iz
        C_T = self.sp.C_T

        # Dynamics
        f = spy.zeros(self.n_x, 1)

        f[0] = self.x[3] * spy.cos(self.x[2]) - self.x[4] * spy.sin(self.x[2])
        f[1] = self.x[3] * spy.sin(self.x[2]) + self.x[4] * spy.cos(self.x[2])
        f[2] = self.x[5]
        f[3] = (1 / self.x[7]) * spy.cos(self.x[6]) * self.u[0] + self.x[5] * self.x[4]
        f[4] = (1 / self.x[7]) * spy.sin(self.x[6]) * self.u[0] - self.x[5] * self.x[3]
        f[5] = (-l_r / I_z) * spy.sin(self.x[6]) * self.u[0]
        f[6] = self.u[1]
        f[7] = -C_T * self.u[0]

        f = self.p[0] * f

        A = f.jacobian(self.x)
        B = f.jacobian(self.u)
        F = f.jacobian(self.p)

        f_func = spy.lambdify((self.x, self.u, self.p), f, "numpy")
        A_func = spy.lambdify((self.x, self.u, self.p), A, "numpy")
        B_func = spy.lambdify((self.x, self.u, self.p), B, "numpy")
        F_func = spy.lambdify((self.x, self.u, self.p), F, "numpy")

        # DEBUGGING
        """print("\nf = ", f)
        print("\nA = ", A)
        print("\nB = ", B)
        print("\nF = ", F)

        print("Symbols in f: ", f.free_symbols)
        print("Symbols in A: ", A.free_symbols)
        print("Symbols in B: ", B.free_symbols)
        print("Symbols in F: ", F.free_symbols)"""

        return f_func, A_func, B_func, F_func
