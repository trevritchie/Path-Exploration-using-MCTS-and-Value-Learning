import math

class MCTSNode:
    """node for monte carlo tree search"""
    def __init__(self, city, parent=None):
        self.city = city
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
        self.unexplored_cities = []

    def add_child(self, child_node):
        """add a child node"""
        self.children.append(child_node)

    def update(self, reward):
        """update node statistics"""
        self.visits += 1
        self.value += reward

    def is_fully_expanded(self):
        """check if all possible children have been created"""
        return len(self.unexplored_cities) == 0

    def best_child(self, exploration_weight=1.0):
        """select best child using UCB1 formula"""
        if not self.children:
            return None

        # UCB1 formula: value/visits + exploration_weight * sqrt(2 * ln(parent_visits) / child_visits)
        best_score = float('-inf')
        best_child = None

        for child in self.children:
            # avoid division by zero
            if child.visits == 0:
                continue

            # exploitation term
            exploitation = child.value / child.visits

            # exploration term
            exploration = exploration_weight * math.sqrt(2 * math.log(self.visits) / child.visits)

            # UCB1 score
            score = exploitation + exploration

            if score > best_score:
                best_score = score
                best_child = child

        return best_child