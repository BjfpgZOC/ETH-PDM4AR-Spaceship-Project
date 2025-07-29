from dataclasses import dataclass, field
from math import inf
from typing import Union

import cvxpy as cvx
from dg_commons import PlayerName
from dg_commons.seq import DgSampledSequence
from dg_commons.sim.models.obstacles_dyn import DynObstacleState
from dg_commons.sim.models.spaceship import SpaceshipCommands, SpaceshipState
from dg_commons.sim.models.spaceship_structures import (
    SpaceshipGeometry,
    SpaceshipParameters,
)

from pdm4ar.exercises.ex11.discretization import *

from pdm4ar.exercises_def.ex11.utils_params import PlanetParams, SatelliteParams
from pdm4ar.exercises_def.ex11.goal import SpaceshipTarget, DockingTarget  # CUSTOM
from typing import Sequence  # CUSTOM
import matplotlib.pyplot as plt  # CUSTOM
from matplotlib.patches import Circle  # CUSTOM
import math  # CUSTOM


@dataclass(frozen=True)
class SolverParameters:
    """
    Definition space for SCvx parameters in case SCvx algorithm is used.
    Parameters can be fine-tuned by the user.
    """

    # Cvxpy solver parameters
    solver: str = "ECOS"  # specify solver to use
    verbose_solver: bool = False  # if True, the optimization steps are shown
    max_iterations: int = 100  # max algorithm iterations

    # SCVX parameters (Add paper reference)
    lambda_nu: float = 1e5  # slack variable weight
    # lambda_nu: float = 1e4  # slack variable weight
    # weight_p: NDArray = field(default_factory=lambda: 10 * np.array([[1.0]]).reshape((1, -1)))  # weight for final time
    weight_p: NDArray = field(default_factory=lambda: 0.1 * np.array([[1.0]]).reshape((1, -1)))  # weight for final time

    weight_u: float = 5  # CUSTOM
    weight_dir: float = 5  # CUSTOM
    W = np.diag([10, 10])  # Weight Matrix, first entry: thrust_weight, second entry: ddelta_weight # CUSTOM

    # tr_radius: float = 5  # initial trust region radius
    tr_radius: float = 10  # initial trust region radius
    min_tr_radius: float = 1e-4  # min trust region radius
    max_tr_radius: float = 100  # max trust region radius
    rho_0: float = 0.0  # trust region 0
    rho_1: float = 0.25  # trust region 1
    rho_2: float = 0.9  # trust region 2
    alpha: float = 2.0  # div factor trust region update
    beta: float = 3.2  # mult factor trust region update

    # Discretization constants
    K: int = 50  # number of discretization steps
    N_sub: int = 5  # used inside ode solver inside discretization
    # stop_crit: float = 1e-5  # Stopping criteria constant
    stop_crit: float = 1e-2  # Stopping criteria constant


from dg_commons.sim.models.obstacles import StaticObstacle


class SpaceshipPlanner:
    """
    Feel free to change anything in this class.
    """

    planets: dict[PlayerName, PlanetParams]
    satellites: dict[PlayerName, SatelliteParams]
    spaceship: SpaceshipDyn
    sg: SpaceshipGeometry
    sp: SpaceshipParameters
    params: SolverParameters

    # Simpy variables
    x: spy.Matrix
    u: spy.Matrix
    p: spy.Matrix

    n_x: int
    n_u: int
    n_p: int

    X_bar: NDArray
    U_bar: NDArray
    p_bar: NDArray

    def __init__(
        self,
        planets: dict[PlayerName, PlanetParams],
        satellites: dict[PlayerName, SatelliteParams],
        sg: SpaceshipGeometry,
        sp: SpaceshipParameters,
        init_state: SpaceshipState,  # CUSTOM
        goal_object: SpaceshipTarget,  # CUSTOM
        obstacles: Sequence[StaticObstacle],  # CUSTOM
    ):
        """
        Pass environment information to the planner.
        """
        self.planets = planets
        self.satellites = satellites
        self.sg = sg
        self.sp = sp
        self.init_state = init_state  # CUSTOM
        self.goal_object = goal_object  # CUSTOM
        self.goal_state = self.goal_object.target
        self.static_obstacles = obstacles  # CUSTOM
        self.sg_buffer = self.sg.l_f + self.sg.l_c + 0.2  # CUSTOM
        self.roh = 0.0  # CUSTOM
        self.optimal_cost = float("inf")  # CUSTOM
        self.fixed_time = 20
        self.current_time = 0.0
        self.time_lower_bound = 5.0

        # Solver Parameters
        self.params = SolverParameters()
        self.eta = self.params.tr_radius  # CUSTOM

        # Spaceship Dynamics
        self.spaceship = SpaceshipDyn(self.sg, self.sp)

        # Discretization Method
        # self.integrator = ZeroOrderHold(self.Spaceship, self.params.K, self.params.N_sub)
        self.integrator = FirstOrderHold(self.spaceship, self.params.K, self.params.N_sub)

        # Variables
        self.variables = self._get_variables()

        # Problem Parameters
        self.problem_parameters = self._get_problem_parameters()

        # Constraints
        # constraints = self._get_constraints()
        self.constraints = self._get_constraints()

        # Objective
        # objective = self._get_objective()
        self.objective = self._get_objective()

    def compute_trajectory(
        self, init_state: SpaceshipState, goal_state: DynObstacleState, current_time=0.0
    ) -> tuple[DgSampledSequence[SpaceshipCommands], DgSampledSequence[SpaceshipState]]:
        """
        Compute a trajectory from init_state to goal_state.
        """
        self.current_time = current_time
        self.fixed_time = self.update_final_time()

        self.init_state = init_state  # update init_state <- current_state (we replan from current_state)
        self.goal_state = goal_state

        # get initial state and trajectory guess guess
        self.X_bar, self.U_bar, self.p_bar = self.initial_guess()

        # for i in range(self.params.max_iterations):
        for i in range(self.params.max_iterations):

            # assign initial guess to decision variables
            self.variables["X"].value = self.X_bar
            self.variables["U"].value = self.U_bar
            self.variables["p"].value = self.p_bar

            # convexify problem parameters around current X_bar, U_bar, p_bar and populate problem parameters
            self._convexification()

            # convexify nonconvex constraints around current X_bar, U_bar, p_bar and add to problem constraints
            constraints = (
                self.constraints
                + self.get_spaceship_dyn_constraints()
                + self.get_trustregion_constraint()
                + self.get_convex_obst_avoid_constraints()
            )

            # update CVXPY problem
            self.problem = cvx.Problem(self.objective, constraints)

            # Solve Optimization Problem (optimal trajectory then is stored in the cvx.Variables)
            try:
                optimal_value = self.problem.solve(verbose=self.params.verbose_solver, solver=self.params.solver)
            except cvx.SolverError:
                print(f"SolverError: {self.params.solver} failed to solve the problem.")

            if type(self.optimal_cost) == float:
                self.optimal_cost = optimal_value

            # Check Convergence (if converged returns True, else updates model accuracy and trust region radius):
            if self._check_convergence():

                # reset eta to default
                self.eta = self.params.tr_radius

                # reformat optimal trajectory for Simulator
                mycmds, mystates = self._extract_seq_from_array()
                return mycmds, mystates

    def initial_guess(self) -> tuple[NDArray, NDArray, NDArray]:
        """
        Define initial guess for SCvx.
        """
        K = self.params.K

        X = np.zeros((self.spaceship.n_x, K))
        U = np.zeros((self.spaceship.n_u, K))
        p = np.zeros((self.spaceship.n_p))

        # Pose:
        current_x, current_y, current_psi = self.init_state.x, self.init_state.y, self.init_state.psi
        goal_x, goal_y, goal_psi = self.goal_state.x, self.goal_state.y, self.goal_state.psi

        x_traj_guess = np.linspace(current_x, goal_x, K)
        y_traj_guess = np.linspace(current_y, goal_y, K)
        dir_traj_guess = np.linspace(current_psi, goal_psi)

        # Final Time
        """dx = goal_x - current_x
        dy = goal_y - current_y
        distance = np.linalg.norm([dx, dy])
        thrust_max = self.sp.thrust_limits[1]
        initial_mass = self.init_state.m
        tf_guess = np.sqrt(2 * initial_mass * distance / (thrust_max / 2))"""
        tf_guess = self.fixed_time

        # Twist:
        v_traj_guess = np.zeros((self.spaceship.n_u, K))
        dpsi_traj_guess = np.zeros(K)

        # Thruster Angle:
        delta_traj_guess = np.zeros(K)

        # Mass:
        m_traj_guess = np.linspace(self.init_state.m, self.sp.m_v, K)

        # Thrust Force:
        # thrust_traj_guess = self.sp.thrust_limits[1] / 2 * np.ones(K)
        # thrust_traj_guess = self.sp.thrust_limits[1] * np.ones(K)
        # t = np.linspace(0, K - 1, K)
        # thrust_traj_guess = 2 * np.sin(3 * np.pi * t / K)
        thrust_traj_guess = np.zeros(K)

        # Thruster Angular Velocity:
        ddelta_traj_guess = np.zeros(K)

        # Populate Reference Trajectory with Initial Guess
        X[0, :] = x_traj_guess
        X[1, :] = y_traj_guess
        X[2, :] = dir_traj_guess
        X[3:5, :] = v_traj_guess
        X[5, :] = dpsi_traj_guess
        X[6, :] = delta_traj_guess
        X[7, :] = m_traj_guess
        U[0, :] = thrust_traj_guess
        U[1, :] = ddelta_traj_guess
        p[0] = tf_guess

        # Virtual Control Inputs:
        self.variables["nu"].value = np.zeros((self.spaceship.n_x, K))
        self.variables["nu_s"].value = np.zeros((len(self.planets) + len(self.satellites), self.params.K))

        return X, U, p

    def _set_goal(self):
        """
        Sets goal for SCvx.
        """

        K = self.params.K
        goal_psi = self.goal_state.psi
        theta = np.deg2rad(30)  # Cone

        postol = 0.4  # 0.1
        dirtol = 0.5  # 0.5
        veltol = 0.5  # 0.5

        if isinstance(self.goal_object, DockingTarget):
            A, B, C, A1, A2, p = self.goal_object.get_landing_constraint_points()
            # A, B, C, A1, A2, p = self.goal_object.get_landing_constraint_points_fix()

            docking_c = []

            # Final Position
            docking_c.append(cvx.norm(self.variables["X"][0:2, -1] - A, 2) <= postol)  # self.goal_object.pos_tol)

            # Endpoint within cone
            ax = self.variables["X"][0:2, -1] - A

            # Glide slope cone
            ab = B - A
            ac = C - A

            cross_ab_x = ab[0] * ax[1] - ab[1] * ax[0]
            cross_ac_x = ac[0] * ax[1] - ac[1] * ax[0]

            docking_c.append(cross_ab_x >= 0)  # spaceship on the right side of AB
            docking_c.append(cross_ac_x <= 0)  # spaceship on the left side of AC

            # Orientation and velocity at the endpoint
            docking_c.append(self.variables["X"][2, -1] <= goal_psi + dirtol)  # self.goal_object.dir_tol)
            docking_c.append(self.variables["X"][2, -1] >= goal_psi - dirtol)  # self.goal_object.dir_tol)

            last_steps = K - 5
            for k in range(last_steps, K - 1):
                docking_c.append(
                    cvx.norm(self.variables["X"][3:5, k + 1] - self.variables["X"][3:5, k], 2)
                    <= veltol  # self.goal_object.vel_tol
                )

            return docking_c

        """if isinstance(self.goal_object, DockingTarget):
            A, B, C, A1, A2, p = self.goal_object.get_landing_constraint_points()

            docking_c = []

            # Final Position
            final_x_c = cvx.abs(self.variables["X"][0, -1] - A[0]) <= self.goal_object.pos_tol
            docking_c.append(final_x_c)

            final_y_c = cvx.abs(self.variables["X"][1, -1] - A[1]) <= self.goal_object.pos_tol
            docking_c.append(final_y_c)

            # Glide slope cone
            ab = B - A
            ac = C - A

            last_steps = K - 5

            for k in range(last_steps, K - 1):
                # approaching cone constraints on angle
                docking_c.append(self.variables["X"][2, k] <= goal_psi + self.goal_object.dir_tol)
                docking_c.append(self.variables["X"][2, k] >= goal_psi - self.goal_object.dir_tol)

                # indirect constraint on velocity
                docking_c.append(
                    cvx.norm(self.variables["X"][0, k + 1] - self.variables["X"][0, k], 2) <= self.goal_object.vel_tol
                )
                docking_c.append(
                    cvx.norm(self.variables["X"][1, k + 1] - self.variables["X"][1, k], 2) <= self.goal_object.vel_tol
                )

                # approaching cone constraints on position
                docking_c.append(
                    ab[0] * (self.variables["X"][1, k] - A[1]) - ab[1] * (self.variables["X"][0, k] - A[0]) >= 0.0
                )
                docking_c.append(
                    (self.variables["X"][0, k] - A[0]) * ac[1] - (self.variables["X"][1, k] - A[1]) * ac[0] >= 0.0
                )

            return docking_c"""

        return []

    def _get_variables(self) -> dict:
        """Define optimisation variables for SCvx"""

        variables = {
            # State, Control, and Parameter variables:
            "X": cvx.Variable((self.spaceship.n_x, self.params.K)),
            "U": cvx.Variable((self.spaceship.n_u, self.params.K)),
            "p": cvx.Variable(self.spaceship.n_p),
            # Virtual control inputs:
            "nu": cvx.Variable((self.spaceship.n_x, self.params.K)),
            "nu_s": cvx.Variable((len(self.planets) + len(self.satellites), self.params.K)),
        }

        return variables

    def _get_problem_parameters(self) -> dict:
        """Define problem parameters for SCvx."""

        K = SolverParameters.K
        n_x = self.spaceship.n_x
        n_u = self.spaceship.n_u
        n_p = self.spaceship.n_p

        # assign value to parameter: param.value = parameter_value
        problem_parameters = {
            # Fixed States
            "init_state": cvx.Parameter(n_x),
            "goal_state": cvx.Parameter(6),
            # Convexified Discrete Dynamics (ALL STORED IN FLATTENED FORM !)
            "A_k": cvx.Parameter((n_x * n_x, K - 1)),
            "B_minus_k": cvx.Parameter((n_x * n_u, K - 1)),
            "B_plus_k": cvx.Parameter((n_x * n_u, K - 1)),
            "F_k": cvx.Parameter((n_x * n_p, K - 1)),
            "r_k": cvx.Parameter((n_x, K - 1)),
            "E_k": cvx.Parameter((n_x * n_x)),
            # Solver Parameters
            "eta": cvx.Parameter(),
        }

        return problem_parameters

    def _get_constraints(self) -> list[cvx.Constraint]:
        """Define constraints for SCvx."""
        constraints = []

        # Initial and # Terminal Boundary Constraints:
        constraints += self.get_boundary_constraints()

        # Convex State Constraints:
        constraints += self.get_convex_state_constraints()

        # Convex Input Constraints:
        constraints += self.get_convex_input_constraints()

        return constraints

    def _get_objective(self) -> Union[cvx.Minimize, cvx.Maximize]:
        """Define objective for SCvx."""

        # Terminal Cost:
        phi = self.params.weight_p @ self.variables["p"]

        # Running Cost: by using cvx.norm(a, 1) nu is minimized element wise (stricter than using e.g. cvx.sum(cvx.norm(self.variables["nu"], axis=0)))
        zeta_nu = self.params.lambda_nu * (cvx.norm(self.variables["nu"], 1) + cvx.norm(self.variables["nu_s"], 1))

        input_cost = 0
        for k in range(self.params.K - 1):
            delta_u = self.variables["U"][:, k + 1] - self.variables["U"][:, k]
            input_cost += cvx.quad_form(delta_u, self.params.W)  # computes squared and weighted 2-norm x'Wx

        zeta_u = self.params.weight_u * input_cost

        dir_cost = 0
        for k in range(self.params.K - 1):
            delta_dir = self.variables["X"][2, k + 1] - self.variables["X"][2, k]
            # delta_dir = cvx.abs(cvx.mod(delta_dir + np.pi, 2 * np.pi) - np.pi)
            dir_cost += cvx.abs(delta_dir)

        zeta_dir = self.params.weight_dir * dir_cost

        # Linearized Augmented Cost Function (Total Cost):
        objective = phi + zeta_nu + zeta_u + zeta_dir

        return cvx.Minimize(objective)

    def _convexification(self):
        """
        Perform convexification step, i.e. Linearization and Discretization
        and populate Problem Parameters.
        """
        # ZOH
        # A_bar, B_bar, F_bar, r_bar = self.integrator.calculate_discretization(self.X_bar, self.U_bar, self.p_bar)
        # FOH
        A_bar, B_plus_bar, B_minus_bar, F_bar, r_bar = self.integrator.calculate_discretization(
            self.X_bar, self.U_bar, self.p_bar
        )

        # create flattened E_k matrix
        n_x = self.spaceship.n_x
        E_k = np.eye(self.spaceship.n_x).reshape(n_x * n_x)

        # Populate Parameters:
        self.problem_parameters["init_state"].value = self.X_bar[:, 0]
        self.problem_parameters["goal_state"].value = self.X_bar[:6, -1]
        self.problem_parameters["A_k"].value = A_bar
        self.problem_parameters["B_plus_k"].value = B_plus_bar
        self.problem_parameters["B_minus_k"].value = B_minus_bar
        self.problem_parameters["F_k"].value = F_bar
        self.problem_parameters["r_k"].value = r_bar
        self.problem_parameters["E_k"].value = E_k
        self.problem_parameters["eta"].value = self.eta  # CHECK

    def _check_convergence(self) -> bool:
        """Check convergence of SCvx."""

        # Reference Trajectory:
        X_bar, U_bar, p_bar = self.X_bar, self.U_bar, self.p_bar

        # CVXPY Trajectory:
        X_star, U_star, p_star = self.variables["X"].value, self.variables["U"].value, self.variables["p"].value

        # Evaluate Linear Augmented Cost Function (LACF) of new Optimal Trajectory
        lacf_opt = self.optimal_cost
        # lacf_opt = self.evaluate_linear_augmented_cost_function()

        # Evaluate Nonlinear Augmented Cost Function (NACF) of Reference Trajectory
        nacf_ref = self.evaluate_nonlinear_augmented_cost_function(X_bar, U_bar, p_bar)

        # Denominator
        denominator = nacf_ref - lacf_opt

        # Convergence Check
        stop_crit_less_cons = np.linalg.norm(p_star - p_bar) + np.max(np.linalg.norm(X_star - X_bar), axis=0)
        # print("stop_crit = ", stop_crit_less_cons)

        # CASE: Converged Solution is dynamically feasible
        if stop_crit_less_cons <= self.params.stop_crit:
            # if denominator <= self.params.stop_crit:

            # update reference trajectory with optimal solution
            self.X_bar = X_star
            self.U_bar = U_star
            self.p_bar = p_star

            return True

        # Converged Solution is NOT dynamically feasible
        else:
            # Evaluate Nonlinear Augmented Cost Function (NACF) of new Optimal Trajectory
            nacf_opt = self.evaluate_nonlinear_augmented_cost_function(X_star, U_star, p_star)
            # Accuracy Measure
            self.roh = (nacf_ref - nacf_opt) / denominator
            self._update_trust_region()

            return False

    def _update_trust_region(self):
        """Update trust region radius."""

        # current trust region radius
        eta = self.problem_parameters["eta"].value

        # limits of trust region radius
        eta_0, eta_1 = self.params.min_tr_radius, self.params.max_tr_radius

        # current model accuracy (roh)
        roh = self.roh
        # parameters
        roh_0, roh_1, roh_2 = self.params.rho_0, self.params.rho_1, self.params.rho_2

        # trust region shrink and growth rate
        beta_sh, beta_gr = self.params.alpha, self.params.beta

        # Case: roh < roh_0
        if roh < roh_0:
            self.eta = max(eta_0, eta / beta_sh)
            # X_bar = X_bar
            # U_bar = U_bar
            # p_bar = p_bar

        # Case: roh_0 <= eta < roh_1
        elif roh_0 <= roh and roh < roh_1:
            self.eta = max(eta_0, eta / beta_sh)
            self.X_bar = self.variables["X"].value
            self.U_bar = self.variables["U"].value
            self.p_bar = self.variables["p"].value

        # Case: roh_1 <= roh < roh_2
        elif roh_1 <= roh and roh < roh_2:
            # self.eta = eta
            self.X_bar = self.variables["X"].value
            self.U_bar = self.variables["U"].value
            self.p_bar = self.variables["p"].value

        # Case: roh_2 < roh
        elif roh_2 <= roh:
            self.eta = min(eta_1, eta * beta_gr)
            self.X_bar = self.variables["X"].value
            self.U_bar = self.variables["U"].value
            self.p_bar = self.variables["p"].value

    # @staticmethod
    def _extract_seq_from_array(self) -> tuple[DgSampledSequence[SpaceshipCommands], DgSampledSequence[SpaceshipState]]:
        """
        Example of how to create a DgSampledSequence from numpy arrays and timestamps.
        """
        K = self.params.K
        p = self.variables["p"].value[0]  # optimal solution of p is the actual final time of the trajectory

        # rescale from normalized to real time
        ts = p * np.linspace(0, 1, K) + self.current_time

        # compute state trajectory
        states = [SpaceshipState(*self.X_bar[:, k]) for k in range(K)]
        mystates = DgSampledSequence[SpaceshipState](timestamps=ts, values=states)

        # compute control input trajectory
        cmds_list = [SpaceshipCommands(f, dd) for f, dd in zip(self.U_bar[0, :], self.U_bar[1, :])]
        mycmds = DgSampledSequence[SpaceshipCommands](timestamps=ts, values=cmds_list)

        return mycmds, mystates

    def get_spaceship_dyn_constraints(self):
        """compute the spaceship dynamics constraint for each timestep k
        (attention, involves state and input values for k and k+1 !)"""
        n_x = self.spaceship.n_x
        n_u = self.spaceship.n_u
        n_p = self.spaceship.n_p
        K = self.params.K

        A_k = self.problem_parameters["A_k"].value.reshape(n_x, n_x, K - 1, order="F")
        B_minus_k = self.problem_parameters["B_minus_k"].value.reshape(n_x, n_u, K - 1, order="F")
        B_plus_k = self.problem_parameters["B_plus_k"].value.reshape(n_x, n_u, K - 1, order="F")
        F_k = self.problem_parameters["F_k"].value.reshape(n_x, n_p, K - 1, order="F")
        r_k = self.problem_parameters["r_k"].value.reshape(n_x, K - 1, order="F")
        E_k = self.problem_parameters["E_k"].value.reshape(n_x, n_x, order="F")

        dynamics_c = [
            self.variables["X"][:, k + 1]
            == A_k[:, :, k] @ self.variables["X"][:, k]
            + B_minus_k[:, :, k] @ self.variables["U"][:, k]
            + B_plus_k[:, :, k] @ self.variables["U"][:, k + 1]
            + F_k[:, :, k] @ self.variables["p"]
            + r_k[:, k]
            + E_k @ self.variables["nu"][:, k]
            for k in range(K - 1)
        ]

        return dynamics_c

    def get_boundary_constraints(self):

        boundary_c = []
        # Initial and Terminal Boundary Condition
        initial_condition_c = self.variables["X"][:, 0] == self.init_state.as_ndarray()
        boundary_c.append(initial_condition_c)

        terminal_condition_c = self.variables["X"][:6, -1] == self.goal_state.as_ndarray()
        boundary_c.append(terminal_condition_c)

        # Case: dock
        if isinstance(self.goal_object, DockingTarget):
            boundary_c += self._set_goal()

        return boundary_c

    def get_convex_state_constraints(self):

        # Map Border Constraints:
        for obstacle in self.static_obstacles:
            if obstacle.shape.geom_type == "LineString":
                map_bounds = obstacle.shape.bounds
                break
        x_min, y_min, x_max, y_max = map_bounds
        map_xmin_c = self.variables["X"][0, :] >= x_min + self.sg_buffer
        map_ymin_c = self.variables["X"][1, :] >= y_min + self.sg_buffer
        map_xmax_c = self.variables["X"][0, :] <= x_max - self.sg_buffer
        map_ymax_c = self.variables["X"][1, :] <= y_max - self.sg_buffer

        # Delta Limits Constraint:
        delta_limit = self.sp.delta_limits
        delta_min_c = delta_limit[0] <= self.variables["X"][6, :]  # Example 1: min thrust =  -2.0
        delta_max_c = delta_limit[1] >= self.variables["X"][6, :]  # Example 1: max thrust =  2.0

        # Mass Constraint:
        mass_min_c = self.variables["X"][7, :] >= self.sp.m_v

        convex_state_c = [map_xmin_c, map_ymin_c, map_xmax_c, map_ymax_c, delta_min_c, delta_max_c, mass_min_c]

        return convex_state_c

    def get_convex_input_constraints(self):

        # thrust force limits:
        thrust_limits = self.sp.thrust_limits
        force_min_c = self.variables["U"][0, :] >= thrust_limits[0]
        force_max_c = self.variables["U"][0, :] <= thrust_limits[1]

        # ddelta limits:
        ddelta_limits = self.sp.ddelta_limits
        ddelta_min_c = self.variables["U"][1, :] >= ddelta_limits[0]
        ddelta_max_c = self.variables["U"][1, :] <= ddelta_limits[1]

        # boundary condition:
        u_ic_c = self.variables["U"][:, 0] == 0.0
        u_tc_c = self.variables["U"][:, -1] == 0.0

        convex_input_c = [force_min_c, force_max_c, ddelta_min_c, ddelta_max_c, u_ic_c, u_tc_c]

        return convex_input_c

    def get_final_time_constraint(self):

        tf_c = self.variables["p"] == self.fixed_time

        return [tf_c]

    def get_convex_obst_avoid_constraints(self):

        buffer = self.sg_buffer  # safety margin (float)

        x_bar = self.X_bar[0, :]  # reference trajectory (2xK)
        y_bar = self.X_bar[1, :]
        p_bar = self.p_bar[0]

        # Planets
        planet_avoidance_c = []
        for idx, (planet_names, planet_params) in enumerate(self.planets.items()):
            c_p = np.array(planet_params.center)  # planet center (2x1)
            r_p = planet_params.radius  # planet radius (float)

            x_p, y_p = c_p[0], c_p[1]

            grad_x_at_ref = -2 * (x_bar - x_p)
            grad_y_at_ref = -2 * (y_bar - y_p)

            r_prime = (
                -((x_bar - x_p) ** 2)
                - (y_bar - y_p) ** 2
                + (r_p + buffer) ** 2
                - grad_x_at_ref * x_bar
                - grad_y_at_ref * y_bar
            )

            s_p_convexified = (
                cvx.multiply(grad_x_at_ref, self.variables["X"][0, :])
                + cvx.multiply(grad_y_at_ref, self.variables["X"][1, :])
                + r_prime
                <= self.variables["nu_s"][idx, :]
            )

            planet_avoidance_c += [s_p_convexified]

        # Satellites
        satellite_avoidance_c = []
        for idx, (satellite_names, satellite_params) in enumerate(self.satellites.items()):
            r_s = satellite_params.radius
            r_orb = satellite_params.orbit_r
            omega_s = satellite_params.omega
            tau_s = satellite_params.tau

            # compute absolute position of satellite center for every time step
            k = np.linspace(0, 1, self.params.K)

            # for replanning, the satellite position have to be moved to their respective position at the current simulation time!
            current_state_time = self.current_time
            current_state_tau = omega_s * current_state_time

            x_s_at_ref = x_p + r_orb * np.cos(omega_s * k * p_bar + tau_s + current_state_tau)
            y_s_at_ref = y_p + r_orb * np.sin(omega_s * k * p_bar + tau_s + current_state_tau)

            grad_x_at_ref = -2 * (x_bar - x_s_at_ref)
            grad_y_at_ref = -2 * (y_bar - y_s_at_ref)

            grad_p_at_ref = (
                -2
                * r_orb
                * omega_s
                * k
                * (
                    (x_bar - x_s_at_ref) * np.sin(omega_s * p_bar * k + tau_s + current_state_tau)
                    - (y_bar - y_s_at_ref) * np.cos(omega_s * p_bar * k + tau_s + current_state_tau)
                )
            )

            r_prime_s = (
                -((x_bar - x_s_at_ref) ** 2)
                - (y_bar - y_s_at_ref) ** 2
                + (r_s + buffer) ** 2
                - grad_x_at_ref * x_bar
                - grad_y_at_ref * y_bar
                - grad_p_at_ref * p_bar
            )

            slack_idx = idx + len(self.planets)

            s_s_convexified = (
                cvx.multiply(grad_x_at_ref, self.variables["X"][0, :])
                + cvx.multiply(grad_y_at_ref, self.variables["X"][1, :])
                + cvx.multiply(grad_p_at_ref, self.variables["p"])
                + r_prime_s
                <= self.variables["nu_s"][slack_idx, :]
            )

            satellite_avoidance_c += [s_s_convexified]

        obstacle_avoidance_c = planet_avoidance_c + satellite_avoidance_c

        return obstacle_avoidance_c

    def get_trustregion_constraint(self):

        K = self.params.K
        eta = self.problem_parameters["eta"].value
        q = 2

        # Define scaling factors (can be adjusted as needed)
        alpha_x, alpha_u, alpha_p = 1.0, 1.0, 1.0  # Scalars for scaling

        # Compute norms (squared 2-norm is denoted by '2' in CVXPY)

        trustregion_c = []
        for k in range(K):
            # Compute norm of deviations from the reference trajectory
            norm_delta_x = cvx.norm(self.variables["X"][:, k] - self.X_bar[:, k], axis=0, p=q)
            norm_delta_u = cvx.norm(self.variables["U"][:, k] - self.U_bar[:, k], axis=0, p=q)
            norm_delta_p = cvx.norm(self.variables["p"] - self.p_bar, p=q)

            tr_c_k = alpha_x * norm_delta_x + alpha_u * norm_delta_u + alpha_p * norm_delta_p <= eta

            trustregion_c.append(tr_c_k)

        return trustregion_c

    def evaluate_linear_augmented_cost_function(self):
        """evaluates the Linear Augmented Cost Function w.t.r. to the optimal trajectory X*, U*, p*
        -> yields the same result as lacf = self.problem.solve()"""

        # Terminal Cost:
        phi = self.params.weight_p @ self.variables["p"].value

        # Running Cost:
        zeta = self.params.lambda_nu * (
            np.linalg.norm(self.variables["nu"].value, 1) + np.linalg.norm(self.variables["nu_s"].value, 1)
        )

        # Linearized Augmented Cost Function (Total Cost):
        lacf = phi + zeta

        return lacf

    def evaluate_nonlinear_augmented_cost_function(self, X, U, p):
        """evaluates the nonlinear augmented cost function at trajectory X, U, p"""

        weight_p, lambda_nu = self.params.weight_p, self.params.lambda_nu

        # ----- ORIGINAL TERMINAL COST (phi_nl) -----
        # Original Boundary Conditions (g)
        g_ic, g_tc = self.evaluate_original_convex_constraints(X, U, p)
        g_pos_part = g_ic + g_tc

        # Original Terminal Cost
        phi_nl = float(weight_p @ p) + lambda_nu * g_pos_part

        # ----- ORIGINAL RUNNING COST (zeta_nl) -----
        # Defect
        defect = self.defect(X, U, p)

        # Original Nonconvex Constraints (s)
        s_p_nc, s_s_nc = self.evaluate_original_nonconvex_constraints(X, U, p)
        s_pos_part = s_p_nc + s_s_nc

        # Original Running Cost
        zeta_nu_nl = lambda_nu * (
            np.sum(np.linalg.norm(defect, axis=0)) + np.sum(s_pos_part)
        )  # + np.linalg.norm(self.variables["U"][0,:].value, 1)

        u_star = self.variables["U"].value
        input_cost_nl = 0
        for k in range(self.params.K - 1):
            delta_u = u_star[:, k + 1] - u_star[:, k]
            input_cost_nl += delta_u.T @ self.params.W @ delta_u  # computes squared and weighted 2-norm u'W u

        zeta_u_nl = self.params.weight_u * input_cost_nl

        X_star = self.variables["X"].value
        dir_cost_nl = 0
        for k in range(self.params.K - 1):
            delta_dir = X_star[2, k + 1] - X_star[2, k]
            dir_cost_nl += abs(delta_dir)

        zeta_dir_nl = self.params.weight_dir * dir_cost_nl

        nacf = phi_nl + zeta_nu_nl + zeta_u_nl + zeta_dir_nl

        return nacf

    def defect(self, X, U, p):
        """computes the defect between caused by linearization of the system dynamics"""

        X_nl = self.integrator.integrate_nonlinear_piecewise(
            X, U, p
        )  # propagate each state x_k in X through psi (flow-map = nonlinear dynamics) to x_k+1
        defect = X - X_nl

        return defect

    def evaluate_original_convex_constraints(self, X, U, p):
        """evaluates the original convex constraints at trajectory X, U, p"""

        # Original Initial Boundary Condition g_ic
        g_ic = np.linalg.norm(
            X[:, 0] - self.problem_parameters["init_state"].value
        )  # is non-zero (!= 0) if constraint is violated

        # Original Terminal Boundary Condition g_tc (is <= 0 if constraint is satisfied, else > 0)
        g_tc = np.linalg.norm(X[:6, -1] - self.problem_parameters["goal_state"].value)

        return g_ic, g_tc  # , g_tc_pos, g_tc_dir, g_tc_vel, g_tc_dpsi

    def evaluate_original_nonconvex_constraints(self, X, U, p):
        """evaluates the original nonconvex constraints at trajectory X, U, p"""

        K = self.params.K
        buffer = self.sg_buffer

        x, y, p = X[0, :], X[1, :], p[0]

        # Original Nonconvex Constraints s

        # Planets
        s_p_nc = []
        for idx, (planet_names, planet_params) in enumerate(self.planets.items()):
            c_p = np.array(planet_params.center)
            r_p = planet_params.radius

            x_p, y_p = c_p[0], c_p[1]

            s_p_nonconvex = (
                -((x - x_p) ** 2) - (y - y_p) ** 2 + (r_p + buffer) ** 2
            )  # is positive (> 0) if constraint is violated (i.e. collision happens)

            # implement functionality of positive-part function []+, s.t. there is a non-zero penalty term if a state is in collision with a planet
            s_p_pos_part = np.maximum(0, s_p_nonconvex)

            s_p_nc.append(s_p_pos_part)

        # Satellites
        s_s_nc = []
        for idx, (satellite_names, satellite_params) in enumerate(self.satellites.items()):
            r_s = satellite_params.radius
            r_orb = satellite_params.orbit_r
            omega_s = satellite_params.omega
            tau_s = satellite_params.tau

            # compute absolute position of satellite center for every time step
            k = np.linspace(0, 1, K)

            # for replanning, the satellite position have to be moved to their respective position at the current simulation time!
            current_state_time = self.current_time
            current_state_tau = omega_s * current_state_time

            x_s = x_p + r_orb * np.cos(omega_s * k * p + tau_s + current_state_tau)
            y_s = y_p + r_orb * np.sin(omega_s * k * p + tau_s + current_state_tau)

            s_s_nonconvex = -((x - x_s) ** 2) - (y - y_s) ** 2 + (r_s + buffer) ** 2

            s_s_pos_part = np.maximum(0, s_s_nonconvex)

            s_s_nc.append(s_s_pos_part)

        return s_p_nc, s_s_nc

    def update_final_time(self):
        # Final Time Constraint
        remaining_time_interval = abs(15.0 - self.current_time)
        delta_t = round((remaining_time_interval / self.params.K) / 0.1) * 0.1
        tf = self.params.K * delta_t

        return tf

    # ------ VISUALIZATION ------
    def plot_trajectory(self, X, U, p, filename_suffix):

        x = X[0, :]
        y = X[1, :]

        thrust = U[0, :]
        ddelta = U[1, :]

        # Check for NaN or Inf in data and handle it
        if np.any(np.isnan(X)) or np.any(np.isnan(U)) or np.any(np.isinf(X)) or np.any(np.isinf(U)):
            # print("Error: Data contains NaN or Inf values.")
            x = np.nan_to_num(x, nan=0.0)  # Replace NaN with 0.0
            y = np.nan_to_num(y, nan=0.0)  # Replace NaN with 0.0
            thrust = np.nan_to_num(thrust, nan=0.0)
            ddelta = np.nan_to_num(ddelta, nan=0.0)

        fig, axs = plt.subplots(2, 1, figsize=(10, 6))

        # Subplot State Trajectory
        axs[0].plot(x, y, marker="*", linestyle="-", color="b", label="Trajectory")
        axs[0].set_title("Position Trajectory")
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("y")
        axs[0].grid(True)
        axs[0].legend()
        axs[0].axis("equal")

        K = self.params.K
        buffer = self.sg_buffer

        # Planets
        for idx, (planet_names, planet_params) in enumerate(self.planets.items()):  # EXCLUDE MAP BORDERS !!!
            c_p = np.array(planet_params.center)
            r_p = planet_params.radius

            circle = Circle((c_p[0], c_p[1]), r_p, color="g", fill=False, linestyle="-", linewidth=2, label="Obstacle")
            axs[0].add_patch(circle)
            axs[0].legend()
            circle = Circle(
                (c_p[0], c_p[1]),
                r_p + buffer,
                color="g",
                fill=False,
                linestyle="--",
                linewidth=2,
                label="Obstacle Buffer",
            )
            axs[0].add_patch(circle)
            axs[0].legend()

        """
        # Satellites
        for idx, (satellite_names, satellite_params) in enumerate(self.satellites.items()):
            r_s = satellite_params.radius
            r_orb = satellite_params.orbit_r
            omega_s = satellite_params.omega
            tau_s = satellite_params.tau

            # compute absolute position of satellite center for every time step
            for k in range(K):

                x_s = c_p[0] + r_orb * np.cos(omega_s * k * p + tau_s)
                y_s = c_p[1] + r_orb * np.sin(omega_s * k * p + tau_s)

                circle = Circle((x_s, y_s), r_s, color='g', fill=False, linestyle='-', linewidth=2, label="Obstacle (Circle)")
                axs[0].add_patch(circle)
                axs[0].legend()
                circle = Circle((x_s, y_s), r_s + buffer, color='g', fill=False, linestyle='--', linewidth=2, label="Obstacle (Circle)")
                axs[0].add_patch(circle)
                axs[0].legend()
        """

        # Subplot Input Trajectory
        timesteps = range(len(thrust))
        axs[1].plot(timesteps, thrust, marker="*", linestyle="-", color="r", label="thrust")
        axs[1].plot(timesteps, ddelta, marker="*", linestyle="-", color="c", label="ddelta")
        axs[1].set_title("Input Trajectory")
        axs[1].set_xlabel("Timestep")
        axs[1].set_ylabel("Value")
        axs[1].grid(True)
        axs[1].legend()

        filename = f"trajectory_plot_{filename_suffix}.png"
        plt.savefig(filename)
        plt.close()
        # print(f"Plot saved as {filename}")

    def plot_time_evolution(self, X, U, p, filename_suffix):
        K = self.params.K  # Number of timesteps
        buffer = self.sg_buffer  # Safety buffer for obstacle radius
        x = X[0, :]  # Trajectory x-coordinates
        y = X[1, :]  # Trajectory y-coordinates
        planet_color = "purple"  # Single color for all planets

        for k in range(K):
            fig, ax = plt.subplots(figsize=(8, 8))  # Create a new figure for each timestep

            # Plot current trajectory point
            ax.plot(
                x[: k + 1], y[: k + 1], marker="o", linestyle="-", color="b", label="Trajectory"
            )  # Plot past trajectory up to timestep k
            ax.scatter(x[k], y[k], color="r", s=100, label="Current Position")  # Current point

            # Plot satellites' current positions and orbits
            for satellite_name, satellite_params in self.satellites.items():
                r_s = satellite_params.radius
                r_orb = satellite_params.orbit_r
                omega_s = satellite_params.omega
                tau_s = satellite_params.tau

                # Current satellite position at timestep k
                x_s = r_orb * np.cos(omega_s * k * p + tau_s)
                y_s = r_orb * np.sin(omega_s * k * p + tau_s)

                # Draw both the satellite's actual radius and buffered radius
                for radius, linestyle, label in [(r_s, "-", f"Satellite {satellite_name}"), (r_s + buffer, "--", None)]:
                    circle = Circle(
                        (x_s, y_s), radius, color="g", fill=False, linestyle=linestyle, linewidth=2, label=label
                    )
                    ax.add_patch(circle)

                # Plot satellite center
                ax.scatter(x_s, y_s, color="k", s=50, label=f"Satellite Center ({satellite_name})")

            # Plot stationary planets
            for planet_name, planet_params in self.planets.items():
                c_p = np.array(planet_params.center)
                r_p = planet_params.radius

                # Draw planet with actual radius and buffered radius
                for radius, linestyle, label in [(r_p, "-", f"Planet {planet_name}"), (r_p + buffer, "--", None)]:
                    circle = Circle(
                        (c_p[0], c_p[1]),
                        radius,
                        color=planet_color,
                        fill=False,
                        linestyle=linestyle,
                        linewidth=2,
                        label=label,
                    )
                    ax.add_patch(circle)

                # Plot planet center
                ax.scatter(c_p[0], c_p[1], color=planet_color, s=50, label=f"Planet Center ({planet_name})")

            # Set axis limits and aspect ratio to be consistent
            ax.set_xlim(-11, 25)
            ax.set_ylim(-11, 11)
            ax.set_aspect("equal", adjustable="box")  # Ensures 1:1 aspect ratio

            ax.set_title(f"Trajectory and Satellites at Timestep {k}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.axis("equal")
            ax.grid(True)
            ax.legend()

            # Save the plot as a PNG
            filename = f"trajectory_satellites_planets_timestep_{k}_{filename_suffix}.png"
            plt.savefig(filename)
            plt.close()
            print(f"Saved plot for timestep {k} as {filename}")
