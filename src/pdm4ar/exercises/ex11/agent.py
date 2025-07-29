from dataclasses import dataclass
from os import error
from socket import CAN_RAW_JOIN_FILTERS
from typing import Sequence

from dg_commons import DgSampledSequence, PlayerName
from dg_commons.sim import SimObservations, InitSimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.obstacles_dyn import DynObstacleState
from dg_commons.sim.models.spaceship import SpaceshipCommands, SpaceshipState
from dg_commons.sim.models.spaceship_structures import SpaceshipGeometry, SpaceshipParameters

from pdm4ar.exercises.ex11.planner import SpaceshipPlanner
from pdm4ar.exercises_def.ex11.goal import SpaceshipTarget, DockingTarget
from pdm4ar.exercises_def.ex11.utils_params import PlanetParams, SatelliteParams

import numpy as np  # CUSTOM
from scipy.interpolate import CubicSpline  # CUSTOM
from decimal import Decimal  # CUSTOM
from datetime import datetime, timedelta  # CUSTOM


@dataclass(frozen=True)
class MyAgentParams:
    """
    You can for example define some agent parameters.
    """

    my_tol: float = 0.1
    pos_tol: float = 1
    dir_tol: float = 1


class SpaceshipAgent(Agent):
    """
    This is the PDM4AR agent.
    Do *NOT* modify this class name
    Do *NOT* modify the naming of the existing methods and input/output types.
    """

    init_state: SpaceshipState
    satellites: dict[PlayerName, SatelliteParams]
    planets: dict[PlayerName, PlanetParams]
    goal_state: DynObstacleState

    cmds_plan: DgSampledSequence[SpaceshipCommands]
    state_traj: DgSampledSequence[SpaceshipState]
    myname: PlayerName
    planner: SpaceshipPlanner
    goal: PlanningGoal
    static_obstacles: Sequence[StaticObstacle]
    sg: SpaceshipGeometry
    sp: SpaceshipParameters

    def __init__(
        self,
        init_state: SpaceshipState,
        satellites: dict[PlayerName, SatelliteParams],
        planets: dict[PlayerName, PlanetParams],
    ):
        """
        Initializes the agent.
        This method is called by the simulator only before the beginning of each simulation.
        Provides the SpaceshipAgent with information about its environment, i.e. planet and satellite parameters and its initial position.
        """
        self.init_state = init_state
        self.satellites = satellites
        self.planets = planets

        self.last_replan_time = 0.0  # CUSTOM
        self.replan_interval = 4  # CUSTOM
        self.tf_lower_bound = 10  # CUSTOM
        self.step_size = 0.1  # CUSTOM

    def on_episode_init(self, init_sim_obs: InitSimObservations):
        """
        This method is called by the simulator only at the beginning of each simulation.
        We suggest to compute here an initial trajectory/node graph/path, used by your planner to navigate the environment.

        Do **not** modify the signature of this method.
        """
        self.myname = init_sim_obs.my_name
        self.sg = init_sim_obs.model_geometry
        self.sp = init_sim_obs.model_params
        self.static_obstacles = init_sim_obs.dg_scenario.static_obstacles  # CUSTOM
        # self.planner = SpaceshipPlanner(planets=self.planets, satellites=self.satellites, sg=self.sg, sp=self.sp)
        assert isinstance(init_sim_obs.goal, SpaceshipTarget | DockingTarget)
        self.goal_state = init_sim_obs.goal.target
        self.goal = init_sim_obs.goal  # CUSTOM

        # Initialize Planner
        self.planner = SpaceshipPlanner(
            planets=self.planets,
            satellites=self.satellites,
            sg=self.sg,
            sp=self.sp,
            init_state=self.init_state,
            goal_object=self.goal,  # CUSTOM
            obstacles=self.static_obstacles,  # CUSTOM
        )

        # generate initial state and control input trajectory, i.e. initial X_bar, U_bar -> store in cmds_plan resp. state_traj
        start_time = 0.0  # CUSTOM
        self.cmds_plan, self.state_traj = self.planner.compute_trajectory(self.init_state, self.goal_state, start_time)

    def get_commands(self, sim_obs: SimObservations) -> SpaceshipCommands:
        """
        This method is called by the simulator at every simulation time step. (0.1 sec)
        We suggest to perform two tasks here:
         - Track the computed trajectory (open or closed loop)
         - Plan a new trajectory if necessary
         (e.g., our tracking is deviating from the desired trajectory, the obstacles are moving, etc.)


        Do **not** modify the signature of this method.
        """

        # issue Control Input (command) to Simulator
        # ZeroOrderHold
        # cmds = self.cmds_plan.at_or_previous(sim_obs.time)
        # FirstOrderHold
        cmds = self.cmds_plan.at_interp(sim_obs.time)
        return cmds
