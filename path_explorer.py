import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import time
from collections import defaultdict

from mcts_node import MCTSNode

# Define city positions
city_positions = {
    "Charleston": (0, 0), "North Charleston": (0, 5), "Summerville": (-5, 10), "Goose Creek": (5, 8), 
    "Moncks Corner": (3, 15), "St. Stephen": (2, 22), "Bonneau": (7, 20), "Eutawville": (-3, 25),
    "Santee": (-2, 32), "Elloree": (-4, 36), "Orangeburg": (-6, 42), "St. Matthews": (-3, 50),
    "Cameron": (-8, 48), "Swansea": (-4, 58), "Gaston": (-2, 64), "Cayce": (1, 68), "Columbia": (2, 72),
    "West Columbia": (0, 72), "Blythewood": (3, 80), "Winnsboro": (1, 88), "Ridgeway": (-2, 92),
    "Great Falls": (-4, 100), "Fort Lawn": (-3, 110), "Lancaster": (-2, 118), "Richburg": (-1, 125),
    "Chester": (-5, 130), "Rock Hill": (-2, 140), "Fort Mill": (1, 148), "Pineville": (3, 152), "Charlotte": (5, 160),
    # New cities
    "Ladson": (1, 9), "Ridgeville": (-5, 17), "St. George": (-6, 24), "Lexington": (-3, 70), "Camden": (6, 72),
    "Kershaw": (1, 110), "Baxter Village": (0, 152)
}

# Define road distances
roads = [
    ("Charleston", "North Charleston", 10), ("North Charleston", "Summerville", 15),
    ("North Charleston", "Goose Creek", 10), ("Summerville", "Moncks Corner", 20), 
    ("Goose Creek", "Moncks Corner", 15), ("Moncks Corner", "St. Stephen", 20),
    ("Moncks Corner", "Bonneau", 15), ("St. Stephen", "Bonneau", 10), ("St. Stephen", "Eutawville", 25),
    ("Bonneau", "Eutawville", 20), ("Eutawville", "Santee", 15), ("Santee", "Elloree", 10),
    ("Elloree", "Orangeburg", 20), ("Orangeburg", "St. Matthews", 15), ("St. Matthews", "Cameron", 10),
    ("Cameron", "Swansea", 20), ("Swansea", "Gaston", 10), ("Gaston", "Cayce", 15), ("Cayce", "Columbia", 5),
    ("Columbia", "West Columbia", 5), ("Columbia", "Blythewood", 20), ("Blythewood", "Winnsboro", 15),
    ("Winnsboro", "Ridgeway", 10), ("Ridgeway", "Great Falls", 15), ("Great Falls", "Fort Lawn", 10),
    ("Fort Lawn", "Lancaster", 15), ("Lancaster", "Richburg", 20), ("Richburg", "Chester", 15),
    ("Chester", "Rock Hill", 20), ("Rock Hill", "Fort Mill", 10), ("Fort Mill", "Pineville", 10),
    # New roads
    ("Pineville", "Charlotte", 10), ("North Charleston", "Ladson", 5), ("Ladson", "Goose Creek", 5),
    ("Summerville", "Ladson", 10), ("Summerville", "Ridgeville", 5), ("Ridgeville", "St. George", 10),
    ("St. George", "Santee", 10), ("St. George", "Eutawville", 5), ("Santee", "St. Matthews", 35),
    ("Swansea", "Lexington", 10), ("Gaston", "Lexington", 5), ("Lexington", "West Columbia", 15),
    ("Columbia", "Camden", 15), ("Camden", "Kershaw", 55), ("Kershaw", "Lancaster", 20), ("Camden", "Blythewood", 10),
    ("Kershaw", "Winnsboro", 25), ("Rock Hill", "Baxter Village", 10), ("Fort Mill", "Baxter Village", 5),
    ("Baxter Village", "Charlotte", 20)
    
]

class PathExplorer:
    """class to handle path exploration using MCTS and value learning"""
    def __init__(self, config):
        self.start_city = config['START_CITY']
        self.end_city = config['END_CITY']
        self.energy_threshold = config['ENERGY_THRESHOLD']
        self.exploration_weight = config['EXPLORATION_WEIGHT']
        self.initial_energy = config['INITIAL_ENERGY']
        self.num_iterations = config['NUM_ITERATIONS']
        self.num_delivery_points = config['NUM_DELIVERY_POINTS']
        self.partially_observable = config['PARTIALLY_OBSERVABLE']
        self.use_shortest_path_metric = config['USE_SHORTEST_PATH_METRIC']
        self.search_method = config['SEARCH_METHOD']

        # rewards configuration
        self.rewards = {
            'shortest_path': config['SHORTEST_PATH_REWARD'],
            'non_shortest_path': config['NON_SHORTEST_PATH_REWARD'],
            'delivery_point': config['DELIVERY_POINT_REWARD'],
            'destination': config['DESTINATION_REWARD'],
            'energy_threshold': config['ENERGY_THRESHOLD_PENALTY']
        }

        # create graph
        self.graph = self.create_graph()

        # find shortest path
        self.shortest_path = self.find_shortest_path()

        # select delivery points
        self.delivery_points = self.select_delivery_points(self.num_delivery_points)

        # track discovered delivery points for partial observability
        self.discovered_delivery_points = set()

        # track node values for learning
        self.node_values = defaultdict(float)

        # statistics
        self.stats = {
            'iterations': 0,
            'successful_paths': 0,
            'failed_paths': 0,
            'total_reward': 0,
            'best_reward': float('-inf'),
            'best_path': None,
            'execution_time': 0,
            # additional metrics for deeper analysis
            'unique_cities_visited': set(),                # all unique cities visited across iterations
            'nodes_created': 0,                            # total MCTS nodes created
            'max_tree_depth': 0,                           # maximum depth reached in the tree
            'convergence_iteration': -1,                   # iteration where best solution was found
            'reward_breakdown': {                          # breakdown of rewards by type
                'shortest_path': 0,
                'non_shortest_path': 0,
                'delivery_points': 0,
                'destination': 0,
                'energy_penalties': 0
            },
            'avg_path_length': 0,                          # average path length for successful paths
            'successful_paths_energy': [],                 # energy levels at termination for successful paths
            'delivery_points_visited': defaultdict(int)    # count of each delivery point being visited
        }

    def create_graph(self):
        """create graph from road data"""
        G = nx.Graph()
        G.add_weighted_edges_from(roads)
        return G

    def find_shortest_path(self):
        """find shortest path using Dijkstra's algorithm"""
        try:
            return nx.shortest_path(self.graph, source=self.start_city,
                                   target=self.end_city, weight="weight")
        except nx.NetworkXNoPath:
            print(f"No path exists between {self.start_city} and {self.end_city}")
            return []

    def select_delivery_points(self, num_points):
        """select random delivery points from cities not in shortest path"""
        # create list of cities not in shortest path
        non_path_cities = [city for city in city_positions.keys()
                          if city not in self.shortest_path
                          and city != self.start_city
                          and city != self.end_city]

        # check if we have enough cities
        if len(non_path_cities) < num_points:
            print(f"Warning: Only {len(non_path_cities)} cities available for delivery points")
            return random.sample(non_path_cities, len(non_path_cities))

        # select random delivery points
        return random.sample(non_path_cities, num_points)

    def is_delivery_point_visible(self, city, current_path):
        """determine if a delivery point is visible based on observability setting"""
        if not self.partially_observable:
            return city in self.delivery_points
        else:
            # in partial observability, delivery point is visible only if it's been discovered
            if city in self.discovered_delivery_points:
                return True

            # check if any neighbor of the city is in the current path
            # if so, "discover" this delivery point
            if city in self.delivery_points:
                neighbors = list(self.graph.neighbors(city))
                for neighbor in neighbors:
                    if neighbor in current_path:
                        self.discovered_delivery_points.add(city)
                        return True
            return False

    def calculate_reward(self, city, energy, path):
        """calculate reward/energy gain for visiting a city"""
        reward = 0

        # if using shortest path metric
        if self.use_shortest_path_metric:
            # if the city is in the shortest path, add reward
            if city in self.shortest_path:
                reward += self.rewards['shortest_path']
            # if not in shortest path, lose points
            else:
                reward += self.rewards['non_shortest_path']  # this is negative

        # check if delivery point is visible based on observability setting
        if self.is_delivery_point_visible(city, path):
            reward += self.rewards['delivery_point']

        # if the city is the destination, add destination reward
        if city == self.end_city:
            reward += self.rewards['destination']

        # energy penalty for termination due to low energy
        if energy + reward <= self.energy_threshold:
            reward += self.rewards['energy_threshold']  # this is negative

        return reward

    def run_search(self):
        """run the selected search method"""
        start_time = time.time()

        if self.search_method == 'mcts':
            best_path, best_reward = self.mcts_search(self.num_iterations, self.initial_energy)
        elif self.search_method == 'random':
            best_path, best_reward = self.random_search(self.num_iterations, self.initial_energy)
        else:
            raise ValueError(f"Unknown search method: {self.search_method}")

        self.stats['execution_time'] = time.time() - start_time
        return best_path, best_reward

    def mcts_search(self, num_iterations, initial_energy):
        """monte carlo tree search for path exploration"""
        # Reset statistics for a new run
        self.stats = {
            'iterations': 0,
            'successful_paths': 0,
            'failed_paths': 0,
            'total_reward': 0,
            'best_reward': float('-inf'),
            'best_path': None,
            'execution_time': 0,
            # additional metrics for deeper analysis
            'unique_cities_visited': set(),
            'nodes_created': 0,
            'max_tree_depth': 0,
            'convergence_iteration': -1,
            'reward_breakdown': {
                'shortest_path': 0,
                'non_shortest_path': 0,
                'delivery_points': 0,
                'destination': 0,
                'energy_penalties': 0
            },
            'avg_path_length': 0,
            'total_path_length': 0,
            'successful_paths_energy': [],
            'delivery_points_visited': defaultdict(int)
        }

        # Reset discovered delivery points for a new run
        if self.partially_observable:
            self.discovered_delivery_points = set()

        root = MCTSNode(self.start_city)
        root.unexplored_cities = list(self.graph.neighbors(self.start_city))
        self.stats['nodes_created'] += 1
        self.stats['unique_cities_visited'].add(self.start_city)

        for i in range(num_iterations):
            # selection and expansion
            node, path, energy = self.select_and_expand(root, initial_energy)

            # simulation
            terminal_reward, terminal_path, terminal_energy = self.simulate(node, path.copy(), energy)

            # backpropagation
            self.backpropagate(node, terminal_reward)

            # update statistics
            self.stats['iterations'] += 1
            self.stats['total_reward'] += terminal_reward

            # update unique cities
            for city in terminal_path:
                self.stats['unique_cities_visited'].add(city)

            # update path length stats
            if terminal_energy > self.energy_threshold and self.end_city in terminal_path:
                self.stats['successful_paths'] += 1
                self.stats['total_path_length'] += len(terminal_path)
                self.stats['successful_paths_energy'].append(terminal_energy)

                # track delivery points visited
                for city in terminal_path:
                    if city in self.delivery_points:
                        self.stats['delivery_points_visited'][city] += 1

                # check tree depth (length of path from root to node)
                depth = 0
                current = node
                while current.parent is not None:
                    depth += 1
                    current = current.parent
                self.stats['max_tree_depth'] = max(self.stats['max_tree_depth'], depth)

                # update best path if this one is better
                if terminal_reward > self.stats['best_reward']:
                    self.stats['best_reward'] = terminal_reward
                    self.stats['best_path'] = terminal_path
                    self.stats['convergence_iteration'] = i + 1
            else:
                self.stats['failed_paths'] += 1

        # calculate average path length
        if self.stats['successful_paths'] > 0:
            self.stats['avg_path_length'] = self.stats['total_path_length'] / self.stats['successful_paths']

        # analyze best path for reward breakdown if it exists
        if self.stats['best_path']:
            self.analyze_path_rewards(self.stats['best_path'])

        return self.stats['best_path'], self.stats['best_reward']

    def random_search(self, num_iterations, initial_energy):
        """random search for path exploration (no MCTS)"""
        # Reset statistics for a new run
        self.stats = {
            'iterations': 0,
            'successful_paths': 0,
            'failed_paths': 0,
            'total_reward': 0,
            'best_reward': float('-inf'),
            'best_path': None,
            'execution_time': 0
        }

        # Reset discovered delivery points for a new run
        if self.partially_observable:
            self.discovered_delivery_points = set()

        for i in range(num_iterations):
            path = [self.start_city]
            energy = initial_energy
            current_city = self.start_city

            # continue until we reach destination, run out of energy, or have no valid moves
            while current_city != self.end_city and energy > self.energy_threshold:
                # get neighbors not in path
                neighbors = [city for city in self.graph.neighbors(current_city)
                             if city not in path]

                if not neighbors:
                    break

                # randomly select next city
                next_city = random.choice(neighbors)
                path.append(next_city)

                # calculate reward and update energy
                reward = self.calculate_reward(next_city, energy, path)
                energy += reward

                # update current city
                current_city = next_city

            # update statistics
            self.stats['iterations'] += 1
            self.stats['total_reward'] += energy

            if energy > self.energy_threshold and self.end_city in path:
                self.stats['successful_paths'] += 1

                # update best path if this one is better
                if energy > self.stats['best_reward']:
                    self.stats['best_reward'] = energy
                    self.stats['best_path'] = path
            else:
                self.stats['failed_paths'] += 1

        return self.stats['best_path'], self.stats['best_reward']

    def select_and_expand(self, node, energy):
        """select a node to expand using UCB1 and expand it"""
        path = [node.city]
        current_energy = energy

        # traverse tree until we reach a leaf node
        while node.is_fully_expanded() and node.children:
            # select best child
            node = node.best_child(self.exploration_weight)

            if node is None:
                break

            # update path and energy
            path.append(node.city)
            reward = self.calculate_reward(node.city, current_energy, path)
            current_energy += reward

            # terminal condition: reaching destination or energy below threshold
            if node.city == self.end_city or current_energy <= self.energy_threshold:
                return node, path, current_energy

        # if node is not fully expanded, expand it
        if not node.is_fully_expanded() and node.unexplored_cities:
            # randomly select unexplored city
            next_city = random.choice(node.unexplored_cities)
            node.unexplored_cities.remove(next_city)

            # create new child node
            child = MCTSNode(next_city, parent=node)
            child.unexplored_cities = [city for city in self.graph.neighbors(next_city)
                                       if city not in path]
            node.add_child(child)
            self.stats['nodes_created'] += 1  # track node creation

            # update path and energy
            path.append(child.city)
            reward = self.calculate_reward(child.city, current_energy, path)
            current_energy += reward

            return child, path, current_energy

        return node, path, current_energy

    def simulate(self, node, path, energy):
        """run a random simulation from the node until terminal state"""
        current_city = node.city
        current_energy = energy
        current_path = path

        # simulate until we reach terminal state
        while current_city != self.end_city and current_energy > self.energy_threshold:
            # get neighbors not already in path
            neighbors = [city for city in self.graph.neighbors(current_city)
                        if city not in current_path]

            # no valid moves
            if not neighbors:
                break

            # randomly select next city
            next_city = random.choice(neighbors)
            current_path.append(next_city)

            # calculate reward and update energy
            reward = self.calculate_reward(next_city, current_energy, current_path)
            current_energy += reward

            # update current city
            current_city = next_city

            # update node values for learning
            self.node_values[current_city] += reward

        return current_energy, current_path, current_energy

    def backpropagate(self, node, reward):
        """backpropagate reward up the tree"""
        while node is not None:
            node.update(reward)
            node = node.parent

    def visualize_graph(self, best_path=None):
        """visualize the graph with paths and delivery points highlighted"""
        plt.figure(figsize=(12, 10))

        # get positions for drawing
        pos = {city: city_positions[city] for city in self.graph.nodes()}

        # draw base graph - all possible roads in light gray
        nx.draw_networkx_edges(self.graph, pos, width=1, alpha=1, edge_color='lightgray')

        # draw nodes (after edges to ensure they're on top)
        nx.draw_networkx_nodes(self.graph, pos, node_size=300, node_color='lightblue')

        # highlight shortest path
        if self.shortest_path:
            path_edges = list(zip(self.shortest_path[:-1], self.shortest_path[1:]))
            nx.draw_networkx_edges(self.graph, pos, edgelist=path_edges,
                                  width=2, edge_color='blue')
            nx.draw_networkx_nodes(self.graph, pos, nodelist=self.shortest_path,
                                  node_size=350, node_color='blue')

        # highlight delivery points - different visualization for partial observability
        if self.partially_observable:
            # discovered delivery points in green
            discovered = [city for city in self.delivery_points if city in self.discovered_delivery_points]
            nx.draw_networkx_nodes(self.graph, pos, nodelist=discovered,
                                  node_size=400, node_color='green')

            # undiscovered delivery points in light green (or not shown if truly hidden)
            undiscovered = [city for city in self.delivery_points if city not in self.discovered_delivery_points]
            nx.draw_networkx_nodes(self.graph, pos, nodelist=undiscovered,
                                  node_size=400, node_color='lightgreen', alpha=0.5)
        else:
            # all delivery points in green for full observability
            nx.draw_networkx_nodes(self.graph, pos, nodelist=self.delivery_points,
                                  node_size=400, node_color='green')

        # highlight best path if available
        if best_path:
            best_path_edges = list(zip(best_path[:-1], best_path[1:]))
            nx.draw_networkx_edges(self.graph, pos, edgelist=best_path_edges,
                                  width=2, edge_color='red', style='dashed')

        # highlight start and end cities
        nx.draw_networkx_nodes(self.graph, pos, nodelist=[self.start_city],
                              node_size=450, node_color='red')
        nx.draw_networkx_nodes(self.graph, pos, nodelist=[self.end_city],
                              node_size=450, node_color='purple')

        # add labels
        nx.draw_networkx_labels(self.graph, pos)

        plt.title(f"Path from {self.start_city} to {self.end_city}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def analyze_path_rewards(self, path):
        """analyze rewards breakdown for a given path"""
        energy = self.initial_energy

        for i in range(1, len(path)):  # skip start city
            city = path[i]

            # shortest path rewards
            if self.use_shortest_path_metric:
                if city in self.shortest_path:
                    self.stats['reward_breakdown']['shortest_path'] += self.rewards['shortest_path']
                else:
                    self.stats['reward_breakdown']['non_shortest_path'] += self.rewards['non_shortest_path']

            # delivery point rewards
            if city in self.delivery_points:
                self.stats['reward_breakdown']['delivery_points'] += self.rewards['delivery_point']

            # destination reward
            if city == self.end_city:
                self.stats['reward_breakdown']['destination'] += self.rewards['destination']

            # calculate energy after this step
            curr_path = path[:i+1]
            reward = self.calculate_reward(city, energy, curr_path)
            energy += reward

            # check for energy threshold penalty
            if energy <= self.energy_threshold:
                self.stats['reward_breakdown']['energy_penalties'] += self.rewards['energy_threshold']
