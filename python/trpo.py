from datetime import datetime
from time import perf_counter
import torch.nn as nn
import numpy as np
import torch
from RaceStrategyEngine.race_environment import RaceStrategyEnvironment, make_env, RacingEvalCallback



if __name__ == "__main__":
    from sb3_contrib import TRPO

    from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack, DummyVecEnv, SubprocVecEnv

    from simulation_parameters_example import drivers, race_state, race_config

    stochastic_starting_compound = False
    stochastic_competitor_strategies = False
    stochastic_positions = False
    stochastic_agent_control = False
    duelling_self_play =  False
    race_perspective = "all_drivers"
    simulation_type = "lap_discrete"
    gamma = 1
    n_stack = 1
    clip_obs = 10

    agent_selected_fields=[
        "driver_position",
        "lap_progress",
        "race_progress",
        "relative_intervals",
        "current_stint",
        "number_of_pit_stops",
        "compound_compliant",
        "regulatory_compliant",
        "delta_to_benchmark_tyre_performance",
    ]
    competitor_selected_fields=[
        "driver_position",
        "lap_progress",
        "race_progress",
        "current_stint",
        "number_of_pit_stops",
        "compound_compliant",
        "regulatory_compliant",
        "delta_to_benchmark_tyre_performance",
    ]

    print(
        f"stochastic_starting_compound: {stochastic_starting_compound}"
        f"\nrace_perspective: {race_perspective}"
        f"\nstochastic_competitor_strategies: {stochastic_competitor_strategies}"
        f"\nstochastic_positions: {stochastic_positions}\n"
        f"stochastic_agent_control: {stochastic_agent_control}\nduelling_self_play: {duelling_self_play}\n"
        f"simulation_type: {simulation_type}"
    )
    eval_env = RaceStrategyEnvironment(
        drivers, race_state, race_config,
        race_perspective=race_perspective,
        simulation_type=simulation_type,
        stochastic_competitor_strategies=stochastic_competitor_strategies,
        stochastic_starting_compound=stochastic_starting_compound,
        stochastic_positions=stochastic_positions,
        stochastic_agent_control=stochastic_agent_control, duelling_self_play=duelling_self_play,
        agent_selected_fields=agent_selected_fields,
        competitor_selected_fields=competitor_selected_fields,
    )
    eval_env = VecFrameStack(
        DummyVecEnv([lambda: eval_env]), n_stack=n_stack
    )  # Same frame stacking!
    eval_env = VecNormalize(
        eval_env, training=False, gamma=gamma, norm_reward=False, clip_obs=clip_obs,
    )  # No training, just normalization
    # check_point = CheckpointCallback(save_freq=1, save_path=f"./dueling_big_network_stack_{n_stack}_silu_best_model_{simulation_type}_{race_perspective}_trpo",save_vecnormalize=True)
    callback = RacingEvalCallback(eval_env, n_eval_episodes=30, eval_freq=10_000,)

    num_processes = 1
    print(f"Creating {num_processes} seperate race environments...")

    env_fns = [
        make_env(
            drivers, race_state, race_config,
            race_perspective=race_perspective,
            simulation_type=simulation_type,
            stochastic_competitor_strategies=stochastic_competitor_strategies,
            stochastic_positions =  stochastic_positions,
            stochastic_starting_compound=stochastic_starting_compound, stochastic_agent_control=stochastic_agent_control,
            duelling_self_play=duelling_self_play, agent_selected_fields=agent_selected_fields, competitor_selected_fields=competitor_selected_fields
        )
        for _ in range(num_processes)
    ]
    env = SubprocVecEnv(env_fns)
    env = VecFrameStack(env, n_stack=n_stack)
    env = VecNormalize(
        env,
        gamma=gamma,
        clip_obs= clip_obs,
        norm_reward=False,
    )
    layers = [512, 512, 512, 512, 512]
    model = TRPO("MlpPolicy", env,
                gamma=gamma,
                normalize_advantage=False,
                n_steps=2048,
                n_critic_updates=55,
                batch_size=512,
                learning_rate=1e-3,
                policy_kwargs=dict(
        net_arch=layers,
        activation_fn= nn.LeakyReLU, 
    ),
                tensorboard_log="./tensorboard_logs/",
                verbose=1,

                  )
    
    

    print(model.policy)
    start_time = datetime.now().strftime("%B %d, %Y at %H:%M:%S")
    print(f"Started learning at: {start_time}")
    print("Default trpo")
    model.learn(total_timesteps=5_000_000* num_processes, callback=callback)


    end_time = datetime.now().strftime("%B %d, %Y at %H:%M:%S") #
    # end_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f"Stopped learning at: {end_time}")

    model.save(f"./TRPO_lap_discrete_model_{layers}_{race_perspective}")
    env.save(f"./TRPO_lap_discrete_norm_{layers}_{race_perspective}")

    print("TESTING")
    stochastic_starting_compound = True
    stochastic_competitor_strategies = False
    stochastic_positions = True
    stochastic_agent_control = False
    duelling_self_play = False
    actual_race_env = RaceStrategyEnvironment(
        drivers, race_state, race_config,
        race_perspective=race_perspective,
        simulation_type=simulation_type,
        stochastic_competitor_strategies=stochastic_competitor_strategies,
        stochastic_starting_compound=stochastic_starting_compound,
        stochastic_positions=stochastic_positions,
        stochastic_agent_control=stochastic_agent_control,
        duelling_self_play=duelling_self_play,
        agent_selected_fields=agent_selected_fields,
        competitor_selected_fields=competitor_selected_fields,
    )

    race_env = VecFrameStack(DummyVecEnv([lambda: actual_race_env]), n_stack=n_stack)
    race_env = VecNormalize(race_env, training=False, gamma=gamma, norm_reward=False, clip_obs=clip_obs)
    # Then copy the trained normalization stats:
    race_env.obs_rms = env.obs_rms
    race_env.ret_rms = env.ret_rms

    positions = []
    strategies = []
    all_pit_stops = []
    compliances = []
    rewards = []
    s = perf_counter()
    obs = race_env.reset()
    n_eval_episodes = 10
    for episode in range(1, n_eval_episodes + 1):
        
        print()
        print("#" * 100)
        print(f"Eval {episode}:")
        print("_"*10, "\n")
        print("Starting Grid")
        print("-" * 50)

        actual_race_env.print_strategies()
        print("-" * 50)

        done = False
        lstm_states = None
        num_envs = 1

        episode_starts = np.ones((num_envs,), dtype=bool)
        prev = None
        lap = 0
        while not done:
            with torch.no_grad():
                # 1. Convert the dict observation to a dict of tensors
                # obs_to_tensor handles the dictionary format automatically
                obs_tensor, _ = model.policy.obs_to_tensor(obs)

                # 2. Get the action distribution from the policy
                distribution = model.policy.get_distribution(obs_tensor)
                
                # 3. Get the probabilities from the distribution
                action_probs_tensor = distribution.distribution.probs
                action_probs_numpy = action_probs_tensor.cpu().numpy().flatten() * 100
                
                # You can now use these probabilities
                # Example action mapping (adjust to your specific action space)
                actions = ["Don't Pit", "Pit for Hard ","Pit for Medium", "Pit for Soft", ]
                
                print("*"*50)
                print(f"Probabilities at the end of lap {lap}:")
                for i, prob in enumerate(action_probs_numpy):
                    # Assuming your actions map 1:1 with the output probabilities
                    print(f"  Action '{actions[i]}': {prob:.4f}%")
                lap += 1
            
            
            
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_starts, deterministic=True
            )
            obs, reward, done, info = race_env.step(action)

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
        positions.append(final_pos)
        strategies.append(strategy)
        all_pit_stops.append(num_pit_stops)
        rewards.append(reward)
        compliances.append(is_regulatory_compliant)

        
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
        print("#" * 100)
        print()

        print()
    # Summary
    avg_position = np.mean(positions)
    avg_pit_stops = np.mean(all_pit_stops)
    avg_rewards = np.mean(rewards)
    best_finish = min(positions)
    worst_finish = max(positions)
    compliance_rate = compliances.count(True) / n_eval_episodes
    top5_rate = sum(1 for p in positions if p <= 5) / n_eval_episodes

    print("=== SUMMARY ===")
    print(f"=== {n_eval_episodes} Evalutions ===")
    print(f"Average Position: P{avg_position:.1f}")
    print(f"Average Reward: {avg_rewards:.1f}")
    print(f"Best Finish: P{best_finish}")
    print(f"Worst Finish: P{worst_finish}")
    print(f"Average Pit Stops: {avg_pit_stops:.1f}")
    print(f"Compliance Rate: {compliance_rate}")
    print(f"Top 5 finishes: {top5_rate}")
    print("=" * 50)
        

    e = perf_counter()
    print(f"Old {e - s}")