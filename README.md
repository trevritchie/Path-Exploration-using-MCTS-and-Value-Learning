# Path Exploration using MCTS and Value Learning
Trevor Ritchie, Carter Quattlebaum, and Donovan Saldarriaga

---
## Description
This project implements path exploration between Charleston, SC and Charlotte, NC using Monte Carlo Tree Search (MCTS) and value learning. The implementation finds the most efficient path between these cities while visiting selected delivery points, managing energy expenditure, and optimizing rewards.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Approach and Design](#approach-and-design)
- [Implementation Process](#implementation-process)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/trevritchie/Path-Exploration-using-MCTS-and-Value-Learning.git 
   ```

2. Navigate to the project directory:

    ```sh
    cd CSCI-470-Classwork/assignment3
    ```

3. No additional dependencies are required beyond standard Python libraries:
    - numpy
    - matplotlib
    - networkx

## Usage

Run the path exploration program with default settings:

```sh
python PathOpt-Learning.py
```

By default, the program runs a single experiment with the configuration specified at the top of the file. For more comprehensive experimentation, we've included a `compare_configs` function that automatically tests different configurations and provides comparative results:

```python
run_experiment()
# compare_configs()
```

The `compare_configs()` function runs a series of tests comparing:
- MCTS vs. random search methods
- Full vs. partial observability of delivery points
- Different initial energy levels (0, 50, 100)

After running all experiments, it prints a summary of results for easy comparison.

You can also modify the configuration variables at the top of `PathOpt-Learning.py` to experiment with different parameters, such as:
- Search method (`SEARCH_METHOD`: 'mcts' or 'random')
- Energy threshold (`ENERGY_THRESHOLD`) and initial energy (`INITIAL_ENERGY`)
- Number of iterations (`NUM_ITERATIONS`)
- Observability settings (`PARTIALLY_OBSERVABLE`: True or False)
- Reward values for different scenarios

## Features

- Graph-based path representation between Charleston and Charlotte
- Shortest path calculation using Dijkstra's algorithm
- Monte Carlo Tree Search (MCTS) implementation for path exploration
- Random search alternative for comparison
- Value learning mechanism to improve path selection
- Energy/reward system with configurable parameters
- Support for both fully observable and partially observable delivery points
- Visualization of paths, delivery points, and exploration results

## Approach and Design

Our approach to implementing path exploration using MCTS and value learning involved several key design decisions:

1. **Graph Representation**: We used a NetworkX graph to represent the cities and roads between Charleston and Charlotte, with edges weighted by road distances. The visualization shows all possible roads as light gray connections, with important paths highlighted in color.

2. **MCTS Implementation**: Our MCTS algorithm follows the standard four steps:
   - Selection: Using UCB1 formula to balance exploration and exploitation
   - Expansion: Adding new nodes to the search tree
   - Simulation: Random rollouts from expanded nodes
   - Backpropagation: Updating node statistics based on simulation results

3. **Reward System**: We implemented a configurable reward system that provides:
   - Positive rewards for cities on the shortest path (15 points)
   - Penalties for cities not on the shortest path (-5 points)
   - Bonuses for delivery points (30 points)
   - Large reward for reaching the destination (100 points)
   - Penalties for depleting energy below threshold (-100 points)

4. **Observability Settings**: We implemented two modes of operation:
   - **Fully Observable** (`PARTIALLY_OBSERVABLE = False`): The agent is aware of all delivery point locations from the beginning, which helps it plan more efficient paths. This knowledge leads to better reward maximization (higher average rewards) and substantially better delivery point coverage (100% vs 40%). Visualization shows the agent making deliberate detours to visit known delivery points.
   - **Partially Observable** (`PARTIALLY_OBSERVABLE = True`): The agent only discovers delivery points when exploring adjacent nodes. This creates a more realistic exploration scenario but results in fewer delivery points visited. The agent must balance exploration (finding unknown delivery points) with exploitation (following the shortest path), leading to slightly higher success rates but lower overall rewards.

5. **Shortest Path Metric**: We implemented a configurable shortest path metric system:
   - When enabled (`USE_SHORTEST_PATH_METRIC = True`): The algorithm rewards staying on the shortest path between Charleston and Charlotte (15 points per node) and penalizes deviations (-5 points). This encourages efficiency while still allowing strategic detours to high-value delivery points.
   - When disabled (`USE_SHORTEST_PATH_METRIC = False`): The algorithm relies solely on energy management without knowledge of the shortest path. This creates a more challenging exploration scenario where the agent must discover efficient routes without guidance.
   - Our experiments showed that using the shortest path metric significantly improved performance, providing the agent with useful guidance that led to more consistent results and higher rewards.

6. **Modularity**: We separated the code into logical components:
   - `PathOpt-Learning.py`: Main configuration and execution
   - `path_explorer.py`: Core path exploration logic
   - `mcts_node.py`: MCTS node implementation

7. **Experimental Insights**: Our experiments revealed several important findings that address the assignment questions:
   - MCTS vs. Random Search: MCTS vastly outperformed random search (98% vs 19.8% success rate), confirming the effectiveness of directed exploration
   - Observability: Full observability of delivery points resulted in better paths, but partial observability still performed well (96.8% success rate)
   - Energy Management: Higher initial energy (100) led to better rewards (615.00) by allowing more flexibility in path selection
   - Convergence: The MCTS algorithm typically found optimal solutions very early (within 2-11 iterations)
   - Reward Structure: The reward distribution showed shortest path rewards dominated the total (375-390 points), suggesting our reward structure effectively guided the algorithm

## Implementation Process

We followed a phased implementation approach:

1. **Foundation Phase**: Created the graph data structure, implemented the shortest path algorithm using Dijkstra's method, set up city data handling, and implemented delivery point selection.

2. **Core MCTS Phase**: Implemented the MCTS algorithm, set up the reward system, handled node selection/expansion, and implemented simulation/rollout functionality.

3. **Value Learning Phase**: Implemented the learning mechanism, set up energy/points tracking, created termination conditions, and implemented backpropagation.

4. **Experimentation Phase**: Created a parameter adjustment system and an automated experimentation framework:
   - The `run_experiments()` function allows for systematic testing of different configurations
   - Performance metrics are tracked for each experiment configuration
   - Results are summarized for easy comparison across experiments
   - Visualization helps interpret the effects of different parameters on path quality

## Results

We conducted several experiments to evaluate the performance of our implementation:

1. **MCTS vs. Random Search**:
   - **MCTS** dramatically outperformed random search:
     - Success rate: 92.8% vs 12.0% for random search
     - Average reward: 441.12 vs 160.92 for random search
     - MCTS consistently achieved 100% delivery point coverage versus random search's highly inconsistent results
     - The directed exploration of MCTS proved essential for finding optimal paths in complex state spaces
     - Path quality: MCTS-generated paths were more efficient, with better balance between shortest path adherence and delivery point coverage

2. **Energy Management Strategies**:
   - **Initial energy of 50** produced the best overall performance:
     - Initial energy 0: 93.6% success rate, 421.17 average reward
     - Initial energy 50: 95.4% success rate, 513.79 average reward
     - Initial energy 100: 93.0% success rate, 489.51 average reward
   - Moderate initial energy (50) created an optimal balance between path flexibility and directed search
   - Too little energy (0) limited exploration options
   - Too much energy (100) sometimes led to overly exploratory behavior that reduced efficiency
   - Energy threshold setting (-50) allowed sufficient exploration while preventing completely inefficient paths

3. **Observability Impact**:
   - **Full observability** and **partial observability** showed interesting tradeoffs:
     - Full observability: 93.6% success rate, 421.17 average reward, 100% delivery point coverage
     - Partial observability: 95.4% success rate, 386.98 average reward, 40% delivery point coverage
   - Full observability led to higher rewards and complete delivery point coverage
   - Partial observability showed slightly higher success rates but with significantly reduced delivery point coverage
   - The MCTS algorithm adapted differently to each scenario - with full observability focusing on maximizing delivery points, while partial observability optimized for path success
   - This confirms that agent knowledge significantly impacts exploration strategy and outcome quality

4. **MCTS Tree Analysis**:
   - Efficient tree construction with approximately 33-35 nodes created per run
   - All 30 cities in the network were explored in each experiment
   - Best solutions were typically found early (iterations 2-11)
   - Maximum tree depth consistently reached 27-28 nodes, indicating full path exploration
   - Average path length was 28 cities, which is very close to the shortest path length (27)

5. **Reward Distribution Analysis**:
   - Shortest path rewards dominated the total (375-390 points)
   - Delivery point rewards contributed significantly (30-60 points)
   - Destination reward (100 points) was consistent across all successful paths
   - Energy threshold penalties were rarely triggered with our configuration

## Technologies Used

- Python 3.x
- NetworkX for graph implementation
- NumPy for numerical operations
- Matplotlib for visualization

## Contributing

Contributions are welcome! Follow these steps:

4. Fork the repo
5. Create a new branch (`git checkout -b feature-name`)
6. Commit your changes (`git commit -m "Added feature"`)
7. Push to the branch (`git push origin feature-name`)
8. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
