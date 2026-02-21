from collections import defaultdict
from typing import Literal, TypedDict

from RaceStrategyEngine.driver import Driver
from RaceStrategyEngine.race_configuration import RaceConfiguration
from RaceStrategyEngine.utility import RaceState
from RaceStrategyEngine.monte_carlo_race_simulations import MonteCarloRaceSimulations
from stable_baselines3.common.callbacks import EvalCallback
import gymnasium as gym
import numpy as np


AvailableFields = Literal[
    "is_agent",
    "driver_name",
    "driver_position",
    "lap_progress",
    "race_progress",
    "interval_ahead",
    "interval_behind",
    "delta_to_leader",
    "relative_intervals",
    "delta_to_benchmark_tyre_performance",
    "current_stint",
    "race_time",
    "lap_times",
    "driver_in_pit_lane",
    "laps_pitted_on",
    "pitted_previous_lap",
    "number_of_pit_stops",
    "different_compounds_used_count",
    "compound_compliant",
    "pit_lane_compliant",
    "regulatory_compliant",
]


class AgentInfo(TypedDict):
    name: str
    start_position: int
    position: int
    race_time: float
    strategy: str
    laps_pitted_on: list[int]
    compounds_used: int
    compliant: bool
    reward: float


class RaceInfo(TypedDict):
    agent: AgentInfo


class RaceStrategyEnvironment(gym.Env):
    def __init__(
        self,
        drivers: dict[str, Driver],
        starting_race_grid: RaceState,
        race_configuration: RaceConfiguration,
        race_perspective: Literal["agent_only", "all_drivers", "graph", "dict"] = "all_drivers",
        stochastic_competitor_strategies: bool = False,
        stochastic_starting_compound: bool = False,
        stochastic_positions: bool = False,
        stochastic_agent_control: bool = False,
        duelling_self_play: bool = False,
        lap_variation_range: tuple[int, int] = (-10, 10),
        simulation_type: Literal["time_discrete", "lap_discrete"] = "lap_discrete",
        track_name="abu_dhabi",
        time_step: float = 1.0,
        action_complexity: Literal["simple", "complex"] = "simple",
        agent_selected_fields: list[AvailableFields]| None = None,
        competitor_selected_fields: list[AvailableFields] | None = None,
        stack_size: int = 1,
        normalize_observations: bool = False,
        norm_stats_inference_only: bool = False,
        normalization_stats_location: str = "",
    ):

        if agent_selected_fields is None:
            agent_selected_fields = [
                "is_agent",
                "driver_position",
                "lap_progress",
                "race_progress",
                "relative_intervals",
                "current_stint",
                "delta_to_benchmark_tyre_performance",
                "number_of_pit_stops",
                "compound_compliant",
                "pit_lane_compliant",
                "regulatory_compliant",
            ]
        
        if competitor_selected_fields is None:
            competitor_selected_fields = [
                "is_agent",
                "driver_position",
                "lap_progress",
                "race_progress",
                "relative_intervals",
                "current_stint",
                "delta_to_benchmark_tyre_performance",
                "number_of_pit_stops",
                "compound_compliant",
                "pit_lane_compliant",
                "regulatory_compliant",
            ]
    

        self._race_sim = MonteCarloRaceSimulations(
            drivers=drivers,
            starting_race_grid=starting_race_grid,
            race_configuration=race_configuration,
            lap_variation_range=lap_variation_range,
        )

        self._simulation_type = simulation_type
        self._lap_variation_range = lap_variation_range
        self._time_step = time_step
        self._race_config = race_configuration
        self._drivers = drivers
        self._stochastic_competitor_strategies = stochastic_competitor_strategies
        self._stochastic_starting_compound = stochastic_starting_compound
        self._stochastic_positions = stochastic_positions
        self._stochastic_agent_control = stochastic_agent_control
        self._track_name = track_name
        self._duelling_self_play = duelling_self_play
        self._action_complexity = action_complexity
        self._agent_selected_fields = agent_selected_fields
        self._competitor_selected_fields = competitor_selected_fields
        self._race_perspective = race_perspective
        self._stack_size = stack_size
        self._normalize_observations = normalize_observations
        self._normalization_stats_location = normalization_stats_location
        self._norm_stats_inference_only = norm_stats_inference_only

        if self._stack_size > 1 and self._race_perspective == "dict":
            raise ValueError(
                "dict race perspective does not currently support stacking"
            )

        self.num_laps = self._race_sim.race_config.num_laps
        self.num_drivers = len(self._race_sim._starting_grid)
        self.points_distribution = self._race_sim.race_config.points_distribution

        # Not creating these now because they come from rust and cant be serialized
        self.env = None

        driver_name = next(iter(self._drivers))
        self._num_compounds = len(self._drivers[driver_name].tyre_models.keys())
        self.num_choices = self._num_compounds + 1  # The one is for None
        if self._action_complexity == "complex":
            self.num_choices = (self.num_laps - 1) * self._num_compounds + 1
        self.action_space = gym.spaces.Discrete(self.num_choices)
        self.starting_position = None

        self._compliance_rewarded = False
        self._previous_lap_progress = None

        field_count = {
            "is_agent": 1,
            "driver_name": 1,
            "driver_position": 1,
            "lap_progress": 1,
            "race_progress": 1,
            "interval_ahead": 1,
            "interval_behind": 1,
            "delta_to_leader": 1,
            "relative_intervals": self.num_drivers,
            "delta_to_benchmark_tyre_performance": 1,
            "current_stint": 2,
            "race_time": 1,
            "lap_times": self.num_laps,
            "driver_in_pit_lane": 1,
            "laps_pitted_on": self.num_laps,
            "pitted_previous_lap": 1,
            "number_of_pit_stops": 1,
            "different_compounds_used_count": 1,
            "compound_compliant": 1,
            "pit_lane_compliant": 1,
            "regulatory_compliant": 1,
        }

        agent_total = 0
        for field in agent_selected_fields:
            agent_total += field_count[field]

        singe_time_step_agent_total = agent_total
        agent_total *= self._stack_size
        if race_perspective == "agent_only":
            inputs = agent_total

            self.single_step_obs_dim = singe_time_step_agent_total

            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(inputs,),
                dtype=np.float32,
            )
        elif race_perspective == "all_drivers":
            num_npc = self.num_drivers - 1
            npc_total = 0
            for field in competitor_selected_fields:
                npc_total += field_count[field]

            singe_time_step_total = singe_time_step_agent_total + (npc_total * num_npc)
            self.single_step_obs_dim = singe_time_step_total
            npc_total *= self._stack_size
            inputs = agent_total + (npc_total * num_npc)

            # driver its controlling

            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(inputs,),
                dtype=np.float32,
            )

        elif race_perspective == "dict":
            self.observation_space = gym.spaces.Dict(
                {
                    "agent": gym.spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(agent_total,),
                        dtype=np.float32,
                    ),
                    "drivers": gym.spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=((npc_total * num_npc),),
                        dtype=np.float32,
                    ),
                }
            )

        elif race_perspective == "graph":
            num_npc = self.num_drivers - 1
            npc_total = 0
            for field in competitor_selected_fields:
                npc_total += field_count[field]

            singe_time_step_total = singe_time_step_agent_total + (npc_total * num_npc)
            self.single_step_obs_dim = singe_time_step_total
            npc_total *= self._stack_size
            inputs = agent_total + (npc_total * num_npc)

            features_per_node = singe_time_step_agent_total  # Since agent == competitor fields


            # TODO Currently using Box but i should change this Graph space
            # Since now im doing all nodes are connected to each other

            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self._stack_size, self.num_drivers, features_per_node),
                dtype=np.float32,
            )



        else:
            raise ValueError(
                "The race perspective must be 'agent_only' or 'all_drivers' or 'dict' or 'graph'"

            )
        
    def _step(self, action) -> tuple[bool, np.ndarray]:
        self._ensure_env_created()
        if self._race_perspective == "graph":
            done, obs = self.env.step_graph(action)
        else:

            done, obs = self.env.step(action)
        return done, obs
    def step(self, action) -> tuple[np.ndarray, float, bool, bool, RaceInfo]:
        done, obs = self._step(action)

        reward = self._calculate_reward(done)
        terminated = done  # race finished
        truncated = False
        info = {}

        self._accumulated_reward += reward

        if done:
            (
                name,
                starting_position,
                final_pos,
                race_time,
                strategy,
                laps_pitted_on,
                compounds_used,
                compliant,
            ) = self._get_agent_results()
            
            end_of_ep_reward = self._accumulated_reward
            laps_pitted_on = list(laps_pitted_on)

            info["agent"] = {
                "name": name,
                "start_position": starting_position,
                "position": final_pos,
                "race_time": race_time,
                "strategy": strategy,
                "laps_pitted_on": laps_pitted_on,
                "compounds_used": compounds_used,
                "compliant": compliant,
                "reward": end_of_ep_reward,
            }

        return obs, reward, terminated, truncated, info
    
    def _reset(self) -> tuple[np.ndarray]:
        self._ensure_env_created()
        if self._race_perspective == "graph":
            obs = self.env.reset_graph()
        else:

            obs = self.env.reset()
        return obs

    def reset(self, *, seed=None, options=None) -> tuple[np.ndarray, RaceInfo]:
        self._compliance_rewarded = False
        self._previous_lap_progress = None
        self._accumulated_reward = 0
        obs = self._reset()
        info = {}

        (
            name,
            starting_position,
            final_position,
            agent_race_time,
            strategy,
            laps_pitted_on,
            compounds_used,
            regulatory_compliant,
        ) = self._get_agent_results()

        info["agent"] = {
            "name": name,
            "start_position": starting_position,
            "position": None,
            "race_time": agent_race_time,
            "strategy": strategy,
            "laps_pitted_on": list(laps_pitted_on),
            "compounds_used": compounds_used,
            "compliant": regulatory_compliant,
            "reward": None,
        }
        return obs, info
    
    def get_action_mask(self):
        self._ensure_env_created()
        return self.env.get_action_mask()
    
    @property
    def current_state_obseravation(self) -> np.ndarray:
        """Get current state for MCTS compatibility"""

        state = self.get_current_state_observation()

        return state
    

    def get_current_state_observation(self) -> np.ndarray:
        
        self._ensure_env_created()

        if self._race_perspective == "graph":
            obs = self.env.get_current_graph_state_observation()
        else:

            obs = self.env.get_current_state_observation()
        return obs
    
    def branch_from_state(self) -> "RaceStrategyEnvironment":
        """Create a new RaceEnvironment instance from current state"""
        # Get current state from Rust env

        self._ensure_env_created()
        current_state = self.env.branch_from_state()  # This returns RaceStrategyEnvironment

        new_env = object.__new__(RaceStrategyEnvironment)

        # Copy ALL existing attributes by reference
        new_env.__dict__ = self.__dict__.copy()

        # Replace the newly created Rust env with the branched state
        new_env.env = current_state

        return new_env
    
    def _clean_branching_variables(self):
        self._ensure_env_created()

        self.env.clean_branching_variables()
    
    def _get_agent_results(self):
        self._ensure_env_created()
        return self.env.get_agent_results()
    
    def _get_num_active_drivers(self) -> int:
        """ The number of drivers still in the race that have not DNF'd

        Returns:
            int: The number of active drivers in the race
        """
        self._ensure_env_created()
        active_drivers = self.env.get_num_active_drivers()

        return active_drivers
    
    def _get_active_drivers_fully_connected_edge_index(self, include_self_loops: bool = False):
        self._ensure_env_created()
        edge_indices = self.env.get_fully_connected_edge_index(include_self_loops)
        return edge_indices
    
    
    def _calculate_reward(self, done: bool) -> float:

        (
            name,
            _start_position,
            position,
            agent_race_time,
            _strategy,
            _laps_pitted_on,
            _compounds_used,
            regulatory_compliant,
        ) = self._get_agent_results()

        reward = 0.0

        reward += self._time_discrete_reward_shaping(regulatory_compliant)

        


        if not done:
            return reward


        if not regulatory_compliant:
            reward = -1_000
            return reward

        time_reward = (
            -agent_race_time / 10
        )  # Scaled it down (5200s becomes 5.2 ) lower race time means less neg

        position_reward = self._position_reward(
            position, self.num_drivers, self.points_distribution
        )

        reward = position_reward + time_reward

        return reward

    def _position_reward(
        self,
        race_position: int,
        total_drivers: int,
        points_distribution: dict[int, int],
    ) -> float:
        # Base reward: higher for better positions, scaled by grid size
        base_reward = ((total_drivers - race_position + 1) / total_drivers) * 100

        # Bonus for top 10 (actual points positions)
        # {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}  # this is per current regs 2024
        top_10_bonus = points_distribution.get(race_position, 0) * 100

        return base_reward + top_10_bonus
    
    def _time_discrete_reward_shaping(self, regulatory_compliant: bool):
        reward = 0.0

        

        if self._simulation_type == "time_discrete":

            if not self._compliance_rewarded and regulatory_compliant:
                reward += 1000
                self._compliance_rewarded = True

            current_lap_progress = self._get_agent_current_lap_progress()
            pit_entry = self._get_pit_lane_entry()

            if self._previous_lap_progress is not None:
                if self._previous_lap_progress <= pit_entry < current_lap_progress:
                    reward += 100

            self._previous_lap_progress = current_lap_progress
        return reward
    
    def _get_agent_current_lap_progress(self) -> float:
        """Get the lap progress of the agent.

        Useful for time discrete simulation when you want to know exactly where the agent
        is in lap, so you could potential applying action masking until the agent is close to the pit lane.

        Returns:
            float: Agent current lap progress

        Note:
            For lap discrete it will always be 0.0 becuase there is no intermediate position, if step you are
            at the transition of completed lap and new lap.
        """
        self._ensure_env_created()
        lap_progress = self.env.get_agent_current_lap_progress()

        return lap_progress

    def _get_agent_current_lap(self) -> int:
        """Get the current lap of the agent.

        Returns:
            int: Agent current lap
        """
        self._ensure_env_created()
        current_lap = self.env.get_agent_current_lap()

        return current_lap

    def _get_pit_lane_entry(self) -> float | None:
        """Get the pit lane entry of the current track

        Returns:
            float: Pit lane entry

        Notes:
            The lap discrete doesnt contain this value as pit stops happen at the end of lap
        """
        self._ensure_env_created()
        pit_lane_entry = self.env.get_pit_lane_entry()

        return pit_lane_entry

    def _get_agent_compliance(self) -> tuple[bool, bool, bool]:
        self._ensure_env_created()
        compound_compliant, pit_lane_compliant, regulatory_compliant = (
            self.env.get_agent_compliance()
        )

        return compound_compliant, pit_lane_compliant, regulatory_compliant
    
    def print_strategies(self):
        self._ensure_env_created()
        self.env.print_strategies()

    def _ensure_env_created(self):
        if not self.env:
            from strategy_engine_core import RaceStrategyEnvironment
            self._race_sim._enable_exp_rust_backend()
            simulation_data, alternate_strategies = (
                self._race_sim._get_alternate_strategies_rust()
            )
            self.env = RaceStrategyEnvironment(
                self._drivers,
                self._race_config,
                self._race_perspective,
                self._action_complexity,
                simulation_data,
                alternate_strategies,
                self._stochastic_competitor_strategies,
                self._stochastic_starting_compound,
                self._stochastic_positions,
                self._stochastic_agent_control,
                self._duelling_self_play,
                self._lap_variation_range,
                self._simulation_type,
                self._track_name,
                self._time_step,
                self._agent_selected_fields,
                self._competitor_selected_fields,
                self._stack_size,
                self._normalize_observations,
                self._norm_stats_inference_only,
                self._normalization_stats_location,
                self.observation_space.shape[0],
                self.single_step_obs_dim
            )


def make_env(
    drivers: dict[str, Driver],
        starting_race_grid: RaceState,
        race_configuration: RaceConfiguration,
        race_perspective: Literal["agent_only", "all_drivers", "dict"] = "all_drivers",
        stochastic_competitor_strategies: bool = False,
        stochastic_starting_compound: bool = False,
        stochastic_positions: bool = False,
        stochastic_agent_control: bool = False,
        duelling_self_play: bool = False,
        lap_variation_range: tuple[int, int] = (-10, 10),
        simulation_type: Literal["time_discrete", "lap_discrete"] = "lap_discrete",
        track_name="abu_dhabi",
        time_step: float = 1.0,
        action_complexity: Literal["simple", "complex"] = "simple",
        agent_selected_fields: list[AvailableFields] | None = None,
        competitor_selected_fields: list[AvailableFields] | None = None,
        stack_size: int = 1,
        normalize_observations: bool = False,
        norm_stats_inference_only: bool = False,
        normalization_stats_location: str = "",
):
    """Factory function to create environment instances"""

    def _init() -> RaceStrategyEnvironment:
        env =RaceStrategyEnvironment(
            drivers=drivers,
            starting_race_grid=starting_race_grid,
            race_configuration=race_configuration,
            race_perspective=race_perspective,
            stochastic_competitor_strategies=stochastic_competitor_strategies,
            stochastic_starting_compound=stochastic_starting_compound,
            stochastic_positions=stochastic_positions,
            stochastic_agent_control=stochastic_agent_control,
            duelling_self_play=duelling_self_play,
            lap_variation_range=lap_variation_range,
            simulation_type=simulation_type,
            track_name=track_name,
            time_step=time_step,
            action_complexity=action_complexity,
            agent_selected_fields=agent_selected_fields,
            competitor_selected_fields=competitor_selected_fields,
            stack_size=stack_size,
            normalize_observations=normalize_observations,
            norm_stats_inference_only=norm_stats_inference_only,
            normalization_stats_location=normalization_stats_location,
        )

        return env

    return _init

class RacingEvalCallback(EvalCallback):
    def _on_step(self) -> bool:
        continue_training = super()._on_step()
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.detailed_evaluation()
        return continue_training

    def detailed_evaluation(self):
        """Custom F1-specific evaluation with detailed race metrics"""

        print(f"\n=== Detailed Evaluation at {self.num_timesteps:,} steps ===")

        positions = []
        strategies = []
        all_pit_stops = []
        compliances = []
        race_times = []
        rewards = []
        all_stint_lengths = defaultdict(list)
        all_stint_counts = defaultdict(list)
        obs = self.eval_env.reset()
        num_eval = (
            self.n_eval_episodes
            if self.n_eval_episodes < 5 and self.n_eval_episodes != 0
            else 5
        )
        for episode in range(1, num_eval + 1):
            print("#" * 100)
            print(f"Eval {episode}:")
            print("_" * 10, "\n")

            print("*" * 50)
            print("Starting Grid")
            self.eval_env.env_method("print_strategies")
            print("*" * 50)
            done = False
            lstm_states = None
            num_envs = 1

            episode_starts = np.ones((num_envs,), dtype=bool)
            while not done:
                action, lstm_states = self.model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=True,
                )

                obs, reward, done, info = self.eval_env.step(action)

            # Extract agent data from info dict
            # im indexing into the list because the wrappers but everything in a list
            agent_data = info[0][
                "agent"
            ]  # info[0] gets first env, ['agent'] gets agent data
            reward = agent_data["reward"]  # reward[0] gets actual reward value
            name = agent_data["name"]
            final_pos = agent_data["position"]
            start_position = agent_data["start_position"]
            race_time = agent_data["race_time"]
            strategy = agent_data["strategy"]
            laps_pitted_on = agent_data["laps_pitted_on"]
            different_compounds_used_count = agent_data["compounds_used"]
            is_regulatory_compliant = agent_data["compliant"]

            laps_pitted_on = list(laps_pitted_on)
            num_pit_stops = len(laps_pitted_on)

            # Store for summary
            if not is_regulatory_compliant:
                final_pos = np.nan  # Invalid finish

            positions.append(final_pos)
            strategies.append(strategy)
            all_pit_stops.append(num_pit_stops)
            rewards.append(reward)
            compliances.append(is_regulatory_compliant)
            race_times.append(race_time)

            stint_count = defaultdict(int)
            stint_lengths = defaultdict(list)
            for compound, stint in strategy:
                stint_count[compound] += 1
                stint_lengths[compound].append(stint)

            # print(stint_count)
            for compound, count in stint_count.items():
                all_stint_counts[compound].append(count)
                compound_stint_lengths = stint_lengths[compound]
                all_stint_lengths[compound].extend(compound_stint_lengths)

            # Detailed episode output
            print(f"  Position: P{final_pos}")
            print(f"  Started Position: P{start_position}")
            print(f"  Controlling: {name}")
            print(f"  Race time: {race_time:.2f}")
            print(f"  Regulation compliance: {is_regulatory_compliant}")
            print(f"  Reward: {reward}")
            print(f"  Number of different compounds used: {different_compounds_used_count}")
            print(f"  Number of pit stops: {num_pit_stops}")
            print(f"  Strategy: {strategy}")
            print(f"  Pitted on laps: {laps_pitted_on}")

            for compound, count in stint_count.items():
                mean_stint_length = np.mean(stint_lengths[compound])
                print(f"  {compound.capitalize()}: Mean Stint Length = {mean_stint_length:.2f}, Number of Stints = {count}")
            print("#" * 100)
            print()

        # Summary

        mean_position = np.nanmean(positions)
        mean_pit_stops = np.mean(all_pit_stops)
        mean_rewards = np.mean(rewards)
        mean_race_time = np.mean(race_times)
        best_finish = np.nanmin(positions)
        worst_finish = np.nanmax(positions)
        compliance_rate = compliances.count(True) / num_eval

        podium = sum(1 for p in positions if p <= 3) / num_eval
        top_5_rate = sum(1 for p in positions if p <= 5) / num_eval
        top_10_rate = sum(1 for p in positions if p <= 10) / num_eval

        self.logger.record("racing_eval/mean_position", mean_position)
        self.logger.record("racing_eval/mean_race_time", mean_race_time)
        self.logger.record("racing_eval/mean_pit_stops", mean_pit_stops)
        self.logger.record("racing_eval/mean_rewards", mean_rewards)
        self.logger.record("racing_eval/best_finish", best_finish)
        self.logger.record("racing_eval/worst_finish", worst_finish)
        self.logger.record("racing_eval/compliance_rate", compliance_rate)
        self.logger.record("racing_eval/podium_finish_rate", podium)
        self.logger.record("racing_eval/top_5_finish_rate", top_5_rate)
        self.logger.record("racing_eval/top_10_finish_rate", top_10_rate)



        print("=== SUMMARY ===")
        print("Race\n---")
        print(f"Average Position: P{mean_position:.1f}")
        print(f"Average Race Time: P{mean_race_time:.1f}")
        print(f"Average Reward: {mean_rewards:.1f}")
        print(f"Best Finish: P{best_finish}")
        print(f"Worst Finish: P{worst_finish}")
        print(f"Average Pit Stops: {mean_pit_stops:.1f}")
        print(f"Compliance Rate: {compliance_rate}")
        print(f"Podium finishes: {podium}")
        print(f"Top 5 finishes: {top_5_rate}")
        print(f"Top 10 finishes: {top_10_rate}")

        print("\nTyres\n---")
        for compound, stints in all_stint_lengths.items():
            compound: str
            mean_stint_length = np.mean(stints)
            stint_count = all_stint_counts[compound]
            mean_stint_count = np.mean(stint_count)
            print(f"{compound.capitalize()}: Mean Stint Length = {mean_stint_length:.2f}, Mean Number of Stints = {mean_stint_count}")
            self.logger.record(f"racing_eval/{compound}_mean_stint_length", mean_stint_length)
            self.logger.record(f"racing_eval/{compound}_mean_number_of_stints", mean_stint_count)
        print("=" * 50)
    
