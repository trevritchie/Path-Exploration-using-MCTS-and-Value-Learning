# PathOpt-Learning.py

import numpy as np
import matplotlib.pyplot as plt

from path_explorer import PathExplorer

# Config
# ----------------------------------------------------------------
# general parameters
START_CITY = 'Charleston'
END_CITY = 'Charlotte'
SEARCH_METHOD = 'mcts'  # 'mcts' or 'random'

# experiment parameters
ENERGY_THRESHOLD = -50
INITIAL_ENERGY = 50
NUM_ITERATIONS = 500
EXPLORATION_WEIGHT = 1.0
NUM_DELIVERY_POINTS = 5
PARTIALLY_OBSERVABLE = False
USE_SHORTEST_PATH_METRIC = True

# reward parameters
SHORTEST_PATH_REWARD = 15
NON_SHORTEST_PATH_REWARD = -5
DELIVERY_POINT_REWARD = 30
DESTINATION_REWARD = 100
ENERGY_THRESHOLD_PENALTY = -100
# ----------------------------------------------------------------

# print current configuration
def print_config():
    print("Current Configuration:")
    print(f"- Search Method: {SEARCH_METHOD}")
    print(f"- Energy Threshold: {ENERGY_THRESHOLD}")
    print(f"- Initial Energy: {INITIAL_ENERGY}")
    print(f"- Iterations: {NUM_ITERATIONS}")
    print(f"- Partially Observable: {PARTIALLY_OBSERVABLE}")
    print(f"- Use Shortest Path Metric: {USE_SHORTEST_PATH_METRIC}")
    print("\nReward Values:")
    print(f"- Shortest Path: {SHORTEST_PATH_REWARD}")
    print(f"- Non-Shortest Path: {NON_SHORTEST_PATH_REWARD}")
    print(f"- Delivery Point: {DELIVERY_POINT_REWARD}")
    print(f"- Destination: {DESTINATION_REWARD}")
    print(f"- Energy Threshold Penalty: {ENERGY_THRESHOLD_PENALTY}")

# Function to run experiment with current configuration
def run_experiment():
    # create config dictionary to pass to PathExplorer
    config = {
        'START_CITY': START_CITY,
        'END_CITY': END_CITY,
        'SEARCH_METHOD': SEARCH_METHOD,
        'ENERGY_THRESHOLD': ENERGY_THRESHOLD,
        'INITIAL_ENERGY': INITIAL_ENERGY,
        'NUM_ITERATIONS': NUM_ITERATIONS,
        'EXPLORATION_WEIGHT': EXPLORATION_WEIGHT,
        'NUM_DELIVERY_POINTS': NUM_DELIVERY_POINTS,
        'PARTIALLY_OBSERVABLE': PARTIALLY_OBSERVABLE,
        'USE_SHORTEST_PATH_METRIC': USE_SHORTEST_PATH_METRIC,
        'SHORTEST_PATH_REWARD': SHORTEST_PATH_REWARD,
        'NON_SHORTEST_PATH_REWARD': NON_SHORTEST_PATH_REWARD,
        'DELIVERY_POINT_REWARD': DELIVERY_POINT_REWARD,
        'DESTINATION_REWARD': DESTINATION_REWARD,
        'ENERGY_THRESHOLD_PENALTY': ENERGY_THRESHOLD_PENALTY
    }

    # create path explorer with current configuration
    explorer = PathExplorer(config)

    print(f"\n{'-'*50}")
    print(f"PATH EXPLORATION: {START_CITY} → {END_CITY}")
    print(f"{'-'*50}")

    # Shortest path and delivery points info
    print(f"\n1. SETUP INFORMATION:")
    print(f"   ╠═ Shortest path length: {len(explorer.shortest_path)} cities")
    print(f"   ╚═ Delivery points: {', '.join(explorer.delivery_points)}")

    # run search with the selected method
    best_path, best_reward = explorer.run_search()

    # Print consolidated performance metrics
    print(f"\n2. PERFORMANCE SUMMARY:")
    print(f"   ╠═ Method: {SEARCH_METHOD.upper()}")
    print(f"   ╠═ Iterations: {explorer.stats['iterations']}")
    print(f"   ╠═ Success rate: {explorer.stats['successful_paths']/explorer.stats['iterations']:.1%}")
    print(f"   ╠═ Average reward: {explorer.stats['total_reward']/explorer.stats['iterations']:.1f}")
    print(f"   ╚═ Execution time: {explorer.stats['execution_time']:.2f} seconds")

    if best_path:
        # Delivery point coverage
        delivery_points_in_best = [city for city in best_path if city in explorer.delivery_points]
        coverage = len(delivery_points_in_best)/len(explorer.delivery_points)

        print(f"\n3. BEST PATH DETAILS:")
        print(f"   ╠═ Best reward: {best_reward:.1f}")
        print(f"   ╠═ Path length: {len(best_path)} cities")
        print(f"   ╠═ Delivery point coverage: {len(delivery_points_in_best)}/{len(explorer.delivery_points)} ({coverage:.0%})")

        # Reward breakdown
        print(f"\n4. REWARD BREAKDOWN:")
        print(f"   ╠═ Shortest path rewards: {explorer.stats['reward_breakdown']['shortest_path']}")
        print(f"   ╠═ Non-shortest path penalties: {explorer.stats['reward_breakdown']['non_shortest_path']}")
        print(f"   ╠═ Delivery point rewards: {explorer.stats['reward_breakdown']['delivery_points']}")
        print(f"   ╚═ Destination reward: {explorer.stats['reward_breakdown']['destination']}")

        # visualize the graph with best path
        explorer.visualize_graph(best_path)
    else:
        print("No successful path found")
        explorer.visualize_graph()

    return explorer.stats

# Example of how to run experiments with different parameters
def compare_configs():
    # Declare global variables at the beginning of the function
    global SEARCH_METHOD, PARTIALLY_OBSERVABLE, INITIAL_ENERGY, ENERGY_THRESHOLD, USE_SHORTEST_PATH_METRIC

    # Store original configuration
    original_configs = {
        'SEARCH_METHOD': SEARCH_METHOD,
        'ENERGY_THRESHOLD': ENERGY_THRESHOLD,
        'INITIAL_ENERGY': INITIAL_ENERGY,
        'PARTIALLY_OBSERVABLE': PARTIALLY_OBSERVABLE,
        'USE_SHORTEST_PATH_METRIC': USE_SHORTEST_PATH_METRIC
    }

    results = {}

    # Example 1: Compare MCTS vs Random Search
    # MCTS Search
    SEARCH_METHOD = 'mcts'
    print("\n\n=== Running MCTS Search ===")
    results['mcts'] = run_experiment()

    # Random Search
    SEARCH_METHOD = 'random'
    print("\n\n=== Running Random Search ===")
    results['random'] = run_experiment()

    # Example 2: Test partial observability
    SEARCH_METHOD = original_configs['SEARCH_METHOD']  # Restore original

    # Fully observable
    PARTIALLY_OBSERVABLE = False
    print("\n\n=== Running with Full Observability ===")
    results['full_obs'] = run_experiment()

    # Partially observable
    PARTIALLY_OBSERVABLE = True
    print("\n\n=== Running with Partial Observability ===")
    results['partial_obs'] = run_experiment()

    # Example 3: Different initial energy levels
    PARTIALLY_OBSERVABLE = original_configs['PARTIALLY_OBSERVABLE']  # Restore original

    energy_levels = [0, 50, 100]
    for energy in energy_levels:
        INITIAL_ENERGY = energy
        print(f"\n\n=== Running with Initial Energy {energy} ===")
        results[f'energy_{energy}'] = run_experiment()

    # Restore original configurations
    SEARCH_METHOD = original_configs['SEARCH_METHOD']
    ENERGY_THRESHOLD = original_configs['ENERGY_THRESHOLD']
    INITIAL_ENERGY = original_configs['INITIAL_ENERGY']
    PARTIALLY_OBSERVABLE = original_configs['PARTIALLY_OBSERVABLE']
    USE_SHORTEST_PATH_METRIC = original_configs['USE_SHORTEST_PATH_METRIC']

    # Print summary of results as a table
    print("\n\n" + "="*70)
    print(" RESULTS SUMMARY ".center(70, "="))
    print("="*70)
    print(f"{'Experiment':<15} | {'Success Rate':^12} | {'Avg Reward':^12} | {'Time (s)':^10} | {'Coverage':^12}")
    print("-"*70)

    for experiment, stats in results.items():
        success_rate = stats['successful_paths'] / stats['iterations'] if stats['iterations'] > 0 else 0
        avg_reward = stats['total_reward'] / stats['iterations'] if stats['iterations'] > 0 else 0

        # Calculate delivery point coverage for this experiment if available
        coverage = "N/A"
        if 'best_path' in stats and 'delivery_points' in stats:
            delivery_points_visited = len([city for city in stats['best_path'] if city in stats['delivery_points']])
            total_delivery_points = len(stats['delivery_points'])
            coverage = f"{delivery_points_visited}/{total_delivery_points} ({delivery_points_visited/total_delivery_points:.0%})"

        print(f"{experiment:<15} | {success_rate:>11.1%} | {avg_reward:>11.1f} | {stats['execution_time']:>9.2f} | {coverage:^12}")

    print("="*70)

# main entry point
if __name__ == "__main__":
    # print configuration
    print_config()

    print("\nPath Exploration using MCTS & Value Learning")
    print("-------------------------------------------")

    # run a single experiment with current configuration
    run_experiment()

    # run a series of experiments with different configurations
    # compare_configs()
