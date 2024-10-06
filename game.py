import numpy as np

class BoardState:
    """
    Represents a state in the game
    """

    def __init__(self):
        """
        Initializes a fresh game state
        """
        self.N_ROWS = 8
        self.N_COLS = 7

        # Initial state as described: [1,2,3,4,5,3,50,51,52,53,54,52]
        self.state = np.array([1, 2, 3, 4, 5, 3, 50, 51, 52, 53, 54, 52])
        self.decode_state = [self.decode_single_pos(d) for d in self.state]

    def update(self, idx, val):
        """
        Updates both the encoded and decoded states
        """
        self.state[idx] = val
        self.decode_state[idx] = self.decode_single_pos(self.state[idx])

    def make_state(self):
        """
        Creates a new decoded state list from the existing state array
        """
        return [self.decode_single_pos(d) for d in self.state]

    def encode_single_pos(self, cr: tuple):
        """
        Encodes a single coordinate (col, row) -> Z

        Input: a tuple (col, row)
        Output: an integer in the interval [0, 55] inclusive
        """
        col, row = cr
        return row * self.N_COLS + col

    def decode_single_pos(self, n: int):
        """
        Decodes a single integer into a coordinate on the board: Z -> (col, row)

        Input: an integer in the interval [0, 55] inclusive
        Output: a tuple (col, row)
        """
        row = n // self.N_COLS
        col = n % self.N_COLS
        return (col, row)

    def is_termination_state(self):
        """
        Checks if the current state is a termination state. A player wins if:
        - White (bottom) moves their ball to the top row (row 7)
        - Black (top) moves their ball to the bottom row (row 0)
        """
        white_ball_pos = self.state[5]  # White ball index
        black_ball_pos = self.state[11]  # Black ball index

        # Decode the positions
        white_ball_coord = self.decode_single_pos(white_ball_pos)
        black_ball_coord = self.decode_single_pos(black_ball_pos)

        # Check if White's ball is on the top row (row 7) or Black's ball is on bottom row (row 0)
        if white_ball_coord[1] == 7:
            return True  # White wins
        if black_ball_coord[1] == 0:
            return True  # Black wins
        
        return False

    def is_valid(self):
        """
        Checks if a board configuration is valid.
        """
        # All positions should be within the range [0, 55]
        if not all(0 <= pos <= 55 for pos in self.state):
            return False

        # There should be no duplicate positions
        if len(set(self.state)) != len(self.state):
            return False

        # Each player must have 5 block pieces and 1 ball piece
        # We assume the given initial state is valid, so no further validation on the piece counts is needed
        return True

class Rules:

    @staticmethod
    def single_piece_actions(board_state, piece_idx):
        """
        Returns the set of possible actions for the given block piece (not holding a ball) at piece_idx
        """
        current_pos = board_state.decode_state[piece_idx]
        col, row = current_pos
        
        # Possible knight moves (dx, dy)
        possible_moves = [
            (col + 2, row + 1), (col + 2, row - 1), 
            (col - 2, row + 1), (col - 2, row - 1),
            (col + 1, row + 2), (col + 1, row - 2),
            (col - 1, row + 2), (col - 1, row - 2)
        ]
        
        valid_moves = set()
        
        # Check if each move is within the bounds of the board and the destination is unoccupied
        for move in possible_moves:
            move_col, move_row = move
            if 0 <= move_col < board_state.N_COLS and 0 <= move_row < board_state.N_ROWS:
                # Encode the new position
                encoded_pos = board_state.encode_single_pos(move)
                
                # Check if the space is unoccupied
                if encoded_pos not in board_state.state:
                    valid_moves.add(encoded_pos)
        
        return valid_moves

    @staticmethod
    def single_ball_actions(board_state, player_idx):
        """
        Returns the set of possible actions for the ball for the given player
        """
        ball_idx = 5 if player_idx == 0 else 11  # The ball is at index 5 for white, 11 for black
        ball_pos = board_state.decode_state[ball_idx]
        col, row = ball_pos
        
        valid_moves = set()
        
        # Ball can pass like a queen in vertical, horizontal, or diagonal directions
        directions = [
            (1, 0), (-1, 0),  # Horizontal
            (0, 1), (0, -1),  # Vertical
            (1, 1), (1, -1), (-1, 1), (-1, -1)  # Diagonal
        ]
        
        for direction in directions:
            dcol, drow = direction
            cur_col, cur_row = col + dcol, row + drow
            
            while 0 <= cur_col < board_state.N_COLS and 0 <= cur_row < board_state.N_ROWS:
                encoded_pos = board_state.encode_single_pos((cur_col, cur_row))
                
                if encoded_pos in board_state.state:
                    break  # Opponent blocks the pass, can't move further
                
                # Only allow passes to same-colored block pieces
                if encoded_pos in board_state.state[player_idx * 6: (player_idx + 1) * 6]:
                    valid_moves.add(encoded_pos)
                
                cur_col += dcol
                cur_row += drow
        
        return valid_moves
    

class GameSimulator:
    """
    Responsible for handling the game simulation
    """

    def __init__(self, players):
        self.game_state = BoardState()
        self.current_round = -1 ## The game starts on round 0; white's move on EVEN rounds; black's move on ODD rounds
        self.players = players

    def run(self):
        """
        Runs a game simulation
        """
        while not self.game_state.is_termination_state():
            ## Determine the round number, and the player who needs to move
            self.current_round += 1
            player_idx = self.current_round % 2
            ## For the player who needs to move, provide them with the current game state
            ## and then ask them to choose an action according to their policy
            action, value = self.players[player_idx].policy( self.game_state.make_state() )
            print(f"Round: {self.current_round} Player: {player_idx} State: {tuple(self.game_state.state)} Action: {action} Value: {value}")

            if not self.validate_action(action, player_idx):
                ## If an invalid action is provided, then the other player will be declared the winner
                if player_idx == 0:
                    return self.current_round, "BLACK", "White provided an invalid action"
                else:
                    return self.current_round, "WHITE", "Black probided an invalid action"

            ## Updates the game state
            self.update(action, player_idx)

        ## Player who moved last is the winner
        if player_idx == 0:
            return self.current_round, "WHITE", "No issues"
        else:
            return self.current_round, "BLACK", "No issues"

    def generate_valid_actions(self, player_idx: int):
        """
        Given a valid state, and a player's turn, generate the set of possible actions that player can take

        player_idx is either 0 or 1

        Input:
            - player_idx, which indicates the player that is moving this turn. This will help index into the
              current BoardState which is self.game_state
        Outputs:
            - a set of tuples (relative_idx, encoded position), each of which encodes an action. The set should include
              all possible actions that the player can take during this turn. relative_idx must be an
              integer on the interval [0, 5] inclusive. Given relative_idx and player_idx, the index for any
              piece in the boardstate can be obtained, so relative_idx is the index relative to current player's
              pieces. Pieces with relative index 0,1,2,3,4 are block pieces that like knights in chess, and
              relative index 5 is the player's ball piece.
        """
        valid_actions = set()
        offset_idx = player_idx * 6  # Either 0 (white) or 6 (black)
        
        # Iterate over each of the 5 block pieces
        for i in range(5):
            piece_idx = offset_idx + i
            piece_actions = Rules.single_piece_actions(self.game_state, piece_idx)
            for action in piece_actions:
                valid_actions.add((i, action))
        
        # Check ball actions (relative index 5)
        ball_actions = Rules.single_ball_actions(self.game_state, player_idx)
        for action in ball_actions:
            valid_actions.add((5, action))
        
        return valid_actions
    
    def validate_action(self, action: tuple, player_idx: int):
        """
        Checks whether or not the specified action can be taken from this state by the specified player

        Inputs:
            - action is a tuple (relative_idx, encoded position)
            - player_idx is an integer 0 or 1 representing the player that is moving this turn
            - self.game_state represents the current BoardState

        Output:
            - if the action is valid, return True
            - if the action is not valid, raise ValueError
        """
        valid_actions = self.generate_valid_actions(player_idx)
        
        if action not in valid_actions:
            raise ValueError(f"Invalid action: {action} for player {player_idx}")
        
        return True
    
    def update(self, action: tuple, player_idx: int):
        """
        Uses a validated action and updates the game board state
        """
        offset_idx = player_idx * 6 ## Either 0 or 6
        idx, pos = action
        self.game_state.update(offset_idx + idx, pos)
