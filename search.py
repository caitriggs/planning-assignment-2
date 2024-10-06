import numpy as np
import queue, heapq
from game import BoardState, GameSimulator, Rules

class Problem:
    """
    This is an interface which GameStateProblem implements.
    You will be using GameStateProblem in your code. Please see
    GameStateProblem for details on the format of the inputs and
    outputs.
    """

    def __init__(self, initial_state, goal_state_set: set):
        self.initial_state = initial_state
        self.goal_state_set = goal_state_set

    def get_actions(self, state):
        """
        Returns a set of valid actions that can be taken from this state
        """
        pass

    def execute(self, state, action):
        """
        Transitions from the state to the next state that results from taking the action
        """
        pass

    def is_goal(self, state):
        """
        Checks if the state is a goal state in the set of goal states
        """
        return state in self.goal_state_set

class GameStateProblem(Problem):

    def __init__(self, initial_board_state, goal_board_state, player_idx):
        """
        player_idx is 0 or 1, depending on which player will be first to move from this initial state.

        Inputs for this constructor:
            - initial_board_state: an instance of BoardState
            - goal_board_state: an instance of BoardState
            - player_idx: an element from {0, 1}

        How Problem.initial_state and Problem.goal_state_set are represented:
            - initial_state: ((game board state tuple), player_idx ) <--- indicates state of board and who's turn it is to move
              ---specifically it is of the form: tuple( ( tuple(initial_board_state.state), player_idx ) )

            - goal_state_set: set([tuple((tuple(goal_board_state.state), 0)), tuple((tuple(goal_board_state.state), 1))])
              ---in otherwords, the goal_state_set allows the goal_board_state.state to be reached on either player 0 or player 1's
              turn.
        """
        super().__init__(tuple((tuple(initial_board_state.state), player_idx)), set([tuple((tuple(goal_board_state.state), 0)), tuple((tuple(goal_board_state.state), 1))]))
        self.sim = GameSimulator(None)
        self.search_alg_fnc = None
        self.set_search_alg()

    def set_search_alg(self, alg=""):
        """
        If you decide to implement several search algorithms, and you wish to switch between them,
        pass a string as a parameter to alg, and then set:
            self.search_alg_fnc = self.your_method
        to indicate which algorithm you'd like to run.
        """
        if alg=='BFS':
            self.search_alg_fnc = self.bfs_search
        elif alg=='ASTAR':
            self.search_alg_fnc = self.a_star_search
        else:
            self.search_alg_fnc = self.bfs_search
        print(f"Using algo: {self.search_alg_fnc()}")

    def get_actions(self, state: tuple):
        """
        From the given state, provide the set possible actions that can be taken from the state

        Inputs: 
            state: (encoded_state, player_idx), where encoded_state is a tuple of 12 integers,
                and player_idx is the player that is moving this turn

        Outputs:
            returns a set of actions
        """
        s, p = state
        np_state = np.array(s)
        self.sim.game_state.state = np_state
        self.sim.game_state.decode_state = self.sim.game_state.make_state()

        return self.sim.generate_valid_actions(p)

    def execute(self, state: tuple, action: tuple):
        """
        From the given state, executes the given action

        The action is given with respect to the current player

        Inputs: 
            state: is a tuple (encoded_state, player_idx), where encoded_state is a tuple of 12 integers,
                and player_idx is the player that is moving this turn
            action: (relative_idx, position), where relative_idx is an index into the encoded_state
                with respect to the player_idx, and position is the encoded position where the indexed piece should move to.
        Outputs:
            the next state tuple that results from taking action in state
        """
        s, p = state
        k, v = action
        offset_idx = p * 6
        return tuple((tuple( s[i] if i != offset_idx + k else v for i in range(len(s))), (p + 1) % 2))

    ## TODO: Implement your search algorithm(s) here as methods of the GameStateProblem.
    ##       You are free to specify parameters that your method may require.
    ##       However, you must ensure that your method returns a list of (state, action) pairs, where
    ##       the first state and action in the list correspond to the initial state and action taken from
    ##       the initial state, and the last (s,a) pair has s as a goal state, and a=None, and the intermediate
    ##       (s,a) pairs correspond to the sequence of states and actions taken from the initial to goal state.
    ##
    ## NOTE: Here is an example of the format:
    ##       [(s1, a1),(s2, a2), (s3, a3), ..., (sN, aN)] where
    ##          sN is an element of self.goal_state_set
    ##          aN is None
    ##          All sK for K=1...N are in the form (encoded_state, player_idx), where encoded_state is a tuple of 12 integers,
    ##              effectively encoded_state is the result of tuple(BoardState.state)
    ##          All aK for K=1...N are in the form (int, int)
    ##
    ## NOTE: The format of state is a tuple: (encoded_state, player_idx), where encoded_state is a tuple of 12 integers
    ##       (mirroring the contents of BoardState.state), and player_idx is 0 or 1, indicating the player that is
    ##       moving in this state.
    ##       The format of action is a tuple: (relative_idx, position), where relative_idx the relative index into encoded_state
    ##       with respect to player_idx, and position is the encoded position where the piece should move to with this action.
    ## NOTE: self.get_actions will obtain the current actions available in current game state.
    ## NOTE: self.execute acts like the transition function.
    ## NOTE: Remember to set self.search_alg_fnc in set_search_alg above.
    ## 

    def is_goal(self, state):
        """
        Checks if the state is a goal state in the set of goal states
        """
        return state in self.goal_state_set

    def bfs_search(self):
        """
        Implements a Breadth-First Search (BFS) to find the shortest sequence of moves
        from the initial state to the goal state.
        """
        # Initialize BFS structures
        frontier = queue.Queue()
        frontier.put((self.initial_state, []))  # Start with initial state and an empty path
        visited = set()
        visited.add(self.initial_state)

        while not frontier.empty():
            current_state, path = frontier.get()

            # Check if we reached the goal
            if self.is_goal(current_state):
                # Return the path that leads to this goal
                return path + [(current_state, None)]  # Append the final state with no action
            
            # Get possible actions for the current state
            possible_actions = self.get_actions(current_state)

            # For each possible action, generate the next state
            for action in possible_actions:
                next_state = self.execute(current_state, action)

                if next_state not in visited:
                    visited.add(next_state)
                    # Add the new state and the action leading to it to the queue
                    frontier.put((next_state, path + [(current_state, action)]))

        return None  # No solution found

    def a_star_search(self):
        """
        Implements the A* Search algorithm to find the optimal sequence of moves from the
        initial state to the goal state.
        """
        # Initialize the priority queue (min-heap) with the initial state and g(n) = 0
        frontier = []
        heapq.heappush(frontier, (0, self.initial_state, []))  # (f(n), state, path)
        
        # A dictionary to store the cost of the cheapest path to a state
        g_cost = {self.initial_state: 0}

        # A set to keep track of visited states
        visited = set()

        while frontier:
            # Get the state with the lowest f(n) value from the priority queue
            _, current_state, path = heapq.heappop(frontier)

            # If we've reached the goal state, return the path
            if self.is_goal(current_state):
                return path + [(current_state, None)]  # Append the final state with no action

            if current_state in visited:
                continue

            visited.add(current_state)

            # Get possible actions for the current state
            possible_actions = self.get_actions(current_state)

            # For each possible action, generate the next state
            for action in possible_actions:
                next_state = self.execute(current_state, action)

                # Calculate the g(n) for the next state
                new_g_cost = g_cost[current_state] + 1  # Every action has a uniform cost of 1

                if next_state not in g_cost or new_g_cost < g_cost[next_state]:
                    g_cost[next_state] = new_g_cost

                    # Calculate the heuristic cost h(n) for the next state
                    h_cost = self.heuristic(next_state)

                    # Calculate the total cost f(n) = g(n) + h(n)
                    f_cost = new_g_cost + h_cost

                    # Push the next state to the priority queue with the updated f(n)
                    heapq.heappush(frontier, (f_cost, next_state, path + [(current_state, action)]))

        return None  # No solution found

    def heuristic(self, state):
        """
        A simple heuristic function that estimates the cost from the current state to the goal state.
        For example, this could be based on the Manhattan distance or the number of misplaced pieces.
        """
        current_board, player_idx = state
        goal_board = tuple(self.goal_state_set)[0][0]  # Get the goal board state

        # Example heuristic: count the number of pieces that are not in their goal position
        mismatches = sum(1 for i in range(len(current_board)) if current_board[i] != goal_board[i])

        return mismatches


if __name__ == '__main__':
    all_reachable_tests = [
        (
            [
                (1,1),(0,1),(2,1),(1,2),(1,0),(1,1),
                (0,0),(2,0),(0,2),(2,2),(3,3),(3,3)
            ],
            set([(0,1),(2,1),(1,2),(1,0)]),
            0
        ),
        (
            [
                (1,1),(0,1),(2,1),(1,2),(1,0),(1,1),
                (0,0),(2,0),(0,2),(2,2),(3,3),(3,3)
            ],
            set([(2,2)]),
            1
        ),
        (
            [
                (1,1),(0,1),(2,1),(1,2),(1,0),(1,1),
                (0,0),(2,0),(0,2),(2,2),(3,3),(0,0)
            ],
            set(),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(0,0),
                (0,1),(2,1),(3,1),(3,2),(2,3),(0,1)
            ],
            set([(2,0),(0,2),(2,2),(0,3)]),
            0
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(2,3),(0,1)
            ],
            set([(0,0),(0,2),(2,2),(0,3)]),
            0
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(0,2),
                (0,1),(2,1),(3,1),(3,2),(2,3),(0,1)
            ],
            set([(0,0),(2,0),(2,2),(0,3)]),
            0
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,2),
                (0,1),(2,1),(3,1),(3,2),(2,3),(0,1)
            ],
            set([(0,0),(2,0),(0,2),(0,3)]),
            0
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(0,3),
                (0,1),(2,1),(3,1),(3,2),(2,3),(0,1)
            ],
            set([(0,0),(2,0),(0,2),(2,2)]),
            0
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(2,3),(0,1)
            ],
            set([(2,1),(3,1),(3,2),(2,3)]),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(2,3),(2,1)
            ],
            set([(0,1),(3,1),(3,2),(2,3)]),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(2,3),(3,1)
            ],
            set([(0,1),(2,1),(3,2),(2,3)]),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(2,3),(3,2)
            ],
            set([(0,1),(2,1),(3,1),(2,3)]),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(2,3),(2,3)
            ],
            set([(0,1),(2,1),(3,1),(3,2)]),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(1,2),(1,2)
            ],
            set([(0,1),(2,1),(3,1),(3,2)]),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(1,2),(0,1)
            ],
            set([(2,1),(3,1),(3,2),(1,2)]),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(1,2),(2,1)
            ],
            set([(0,1),(3,1),(3,2),(1,2)]),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(1,2),(3,1)
            ],
            set([(0,1),(2,1),(3,2),(1,2)]),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(1,2),(3,2)
            ],
            set([(0,1),(2,1),(3,1),(1,2)]),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(0,0),
                (0,1),(2,1),(3,1),(3,2),(1,2),(3,2)
            ],
            set([(2,0),(0,2),(2,2),(0,3)]),
            0
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(1,2),(3,2)
            ],
            set([(0,0),(0,2),(2,2),(0,3)]),
            0
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(0,2),
                (0,1),(2,1),(3,1),(3,2),(1,2),(3,2)
            ],
            set([(0,0),(2,0),(2,2),(0,3)]),
            0
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,2),
                (0,1),(2,1),(3,1),(3,2),(1,2),(3,2)
            ],
            set([(0,0),(2,0),(0,2),(0,3)]),
            0
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(0,3),
                (0,1),(2,1),(3,1),(3,2),(1,2),(3,2)
            ],
            set([(0,0),(2,0),(0,2),(2,2)]),
            0
        ),
    ]

    def test_ball_reachability(state, reachable, player):
        board = BoardState()
        board.state = np.array(list(board.encode_single_pos(cr) for cr in state))
        board.decode_state = board.make_state()
        predicted_reachable_encoded = Rules.single_ball_actions(board, player)
        encoded_reachable = set(board.encode_single_pos(cr) for cr in reachable)
        print("Predicted reachable: ", predicted_reachable_encoded)
        print("Actual reachable: ", encoded_reachable)
        assert predicted_reachable_encoded == encoded_reachable

    for idx, test in enumerate(all_reachable_tests):
        state = test[0]
        reachable = test[1]
        player = test[2]
        print(f"Test #{idx}")
        print("State: ", state)
        print("Reachable: ", reachable)
        print("Player: ", player)
        print(test_ball_reachability(state, reachable, player))
        print(f"{'='*20}")