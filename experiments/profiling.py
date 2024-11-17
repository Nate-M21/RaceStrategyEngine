from RaceStrategyEngine.monte_carlo_race_simulations import MonteCarloRaceSimulations
from simulation_parameters_example import drivers, race_state, race_config
import time
import cProfile
import pstats
import io
from pstats import SortKey

# Initialize the Monte Carlo race simulation
test_race = MonteCarloRaceSimulations(drivers=drivers, starting_race_grid=race_state, race_configuration=race_config)


def profile_function(func, *args, **kwargs):
    pr = cProfile.Profile()
    pr.enable()
    result = func(*args, **kwargs)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(SortKey.CUMULATIVE)
    ps.print_stats(20)  # Print top 20 time-consuming functions
    print(s.getvalue())
    return result

print("Non-adaptive Monte Carlo simulation:")
profile_function(test_race.run_monte_carlo_simulation)

print("\nAdaptive Monte Carlo simulation:")
profile_function(test_race.run_adaptive_monte_carlo_simulation)

print("\nTesting without cprofil None Adaptive Monte Carlo simulation:")
start_time = time.perf_counter()
test_race.run_monte_carlo_simulation()
end_time = time.perf_counter()
duration = end_time - start_time
print(f"Execution time: {duration} seconds.")

print("\nTesting without cprofil Adaptive Monte Carlo simulation:")
start_time = time.perf_counter()
test_race.run_adaptive_monte_carlo_simulation()
end_time = time.perf_counter()
duration = end_time - start_time
print(f"Execution time: {duration} seconds.")