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
        Checks if the current state is a termination state.
        White wins if their ball is in row 7 (positions 49-55).
        Black wins if their ball is in row 0 (positions 0-6).
        Both players cannot be in a winning state at the same time.
        """
        white_ball_idx = 5  # White's ball is at index 5
        black_ball_idx = 11  # Black's ball is at index 11

        white_wins = 49 <= self.state[white_ball_idx] <= 55  # White wins if in positions 49-55
        black_wins = 0 <= self.state[black_ball_idx] <= 6    # Black wins if in positions 0-6

        # Both players cannot win at the same time
        if white_wins and black_wins:
            return False  # Invalid state: both players can't win simultaneously

        # Return True if either White or Black wins
        return white_wins or black_wins

    def is_valid(self):
        """
        Checks if a board configuration is valid based on the game rules:
        - All pieces must be within the 8x7 grid (positions 0-55).
        - No two block pieces can occupy the same position (balls are allowed to overlap with block pieces).
        - White's ball (index 5) must be on one of White's pieces (indices 0-4).
        - Black's ball (index 11) must be on one of Black's pieces (indices 6-10).
        
        Returns:
            True if the board configuration is valid, otherwise False.
        """
        print(self.state)

        # 1. Check if all positions are within the valid range (0 to 55)
        for pos in self.state:
            if not (0 <= pos <= 55):
                return False

        # 2. Check for overlapping block pieces, excluding ball positions (5 and 11)
        white_pieces = list(self.state[:5].copy())
        black_pieces = list(self.state[6:11].copy())
        block_pieces = white_pieces + black_pieces # Block pieces only, excluding balls
        if len(set(block_pieces)) < len(block_pieces):
            return False  # Invalid state due to overlapping block pieces

        # 3. Check that the white ball (index 5) is on a white block piece (indices 0-4)
        white_ball_pos = self.state[5]
        if white_ball_pos not in self.state[:5]:  # White ball must be on one of White's block pieces
            return False

        # 4. Check that the black ball (index 11) is on a black block piece (indices 6-10)
        black_ball_pos = self.state[11]
        if black_ball_pos not in self.state[6:11]:  # Black ball must be on one of Black's block pieces
            return False

        # If all checks pass, the board configuration is valid
        return True


class Rules:

    @staticmethod
    def single_piece_actions(board_state, piece_idx: int):
        """
        Returns possible L-shaped moves (knight-like) for a block piece at a specific index.
        The piece can only move to unoccupied spaces, and only if it is not holding the ball.

        Input: 
            - board_state: The current state of the game.
            - piece_idx: Index of the block piece (0 to 4 or 6 to 10, since we're excluding the ball).
        
        Output:
            - Set of possible encoded positions (integers) where the block piece can move.
        """
        current_pos = board_state.decode_state[piece_idx]  # Get current (col, row) of the block piece
        col, row = current_pos

        # Check if this piece is holding the ball (no movement allowed if holding the ball)
        ball_idx = 5 if piece_idx < 6 else 11
        if board_state.state[ball_idx] == board_state.state[piece_idx]:
            return set()  # Return an empty set since the block is holding the ball

        # Possible knight-like moves (L-shaped moves)
        moves = [
            (col + 2, row + 1), (col + 2, row - 1), 
            (col - 2, row + 1), (col - 2, row - 1),
            (col + 1, row + 2), (col + 1, row - 2), 
            (col - 1, row + 2), (col - 1, row - 2)
        ]

        valid_moves = set()
        for move in moves:
            new_col, new_row = move

            # Ensure the move is within bounds of the 8x7 grid
            if 0 <= new_col < board_state.N_COLS and 0 <= new_row < board_state.N_ROWS:
                encoded_move = board_state.encode_single_pos(move)

                # Ensure the destination is unoccupied by any piece
                if encoded_move not in board_state.state:
                    valid_moves.add(encoded_move)

        return valid_moves

    @staticmethod
    def single_ball_actions(board_state, player_idx: int):
        """
        Returns possible ball passing actions for a player. The ball can only be passed from one block
        piece to another of the same color along vertical, horizontal, or diagonal paths, provided that
        no opposing block pieces intercept the path.

        Input:
            - board_state: The current state of the game.
            - player_idx: Index of the player (0 for White, 1 for Black).
        
        Output:
            - Set of possible encoded positions (integers) where the ball can be passed to.
        """
        ball_idx = 5 if player_idx == 0 else 11  # White's ball is index 5, Black's ball is index 11
        current_pos = board_state.decode_state[ball_idx]  # Get ball's current position
        col, row = current_pos

        # Get the positions of all block pieces
        own_pieces = [board_state.decode_state[i] for i in range(player_idx * 6, player_idx * 6 + 5)]
        opposing_pieces = [board_state.decode_state[i] for i in range((1 - player_idx) * 6, (1 - player_idx) * 6 + 5)]

        valid_moves = set()

        # Check vertical, horizontal, and diagonal passing channels
        directions = [(0, 1), (1, 0), (1, 1), (-1, 1), (0, -1), (-1, 0), (-1, -1), (1, -1)]

        for dcol, drow in directions:
            new_col, new_row = col + dcol, row + drow

            # Move along the direction until we hit a boundary or a block
            while 0 <= new_col < board_state.N_COLS and 0 <= new_row < board_state.N_ROWS:
                encoded_pos = board_state.encode_single_pos((new_col, new_row))

                # Check if an opposing piece blocks the path
                if (new_col, new_row) in opposing_pieces:
                    break  # Stop if an opponent's piece is in the way

                # If a piece of the same color is at this position, it's a valid passing destination
                if (new_col, new_row) in own_pieces:
                    valid_moves.add(encoded_pos)

                # Move to the next square in the same direction
                new_col += dcol
                new_row += drow

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
        Enumerates all valid actions for a player. An action is encoded as a tuple
        (relative_idx, encoded_position), where relative_idx is the relative index
        of the piece with respect to the player and encoded_position is the new position.
        
        Input:
            - player_idx: Index of the player (0 for White, 1 for Black).
        
        Output:
            - Set of valid actions the player can take.
        """
        actions = set()

        # Generate actions for block pieces (relative indices 0-4 for both players)
        for i in range(5):
            piece_idx = player_idx * 6 + i  # Get actual index in the state array
            valid_moves = Rules.single_piece_actions(self.game_state, piece_idx)
            for move in valid_moves:
                actions.add((i, move))  # (relative index, encoded position)

        # Generate actions for the ball (relative index 5)
        ball_moves = Rules.single_ball_actions(self.game_state, player_idx)
        for move in ball_moves:
            actions.add((5, move))  # (relative index, encoded position)

        return actions

    def validate_action(self, action: tuple, player_idx: int):
        """
        Validates the specified action for a given player. If the action is valid, return True.
        If the action is invalid, raise a ValueError with a description of why the action is not valid.
        
        Input:
            - action: Tuple (relative_idx, encoded_position), where relative_idx is the index
                    relative to the player's pieces and encoded_position is the target position.
            - player_idx: Index of the player (0 for White, 1 for Black).
        
        Output:
            - True if the action is valid, raises ValueError otherwise.
        """
        relative_idx, encoded_pos = action

        if not (0 <= relative_idx <= 5):  # Invalid piece index
            raise ValueError(f"Invalid piece index: {relative_idx}")

        if relative_idx < 5:  # Block pieces (indices 0-4)
            valid_moves = Rules.single_piece_actions(self.game_state, player_idx * 6 + relative_idx)
        else:  # Ball (index 5)
            valid_moves = Rules.single_ball_actions(self.game_state, player_idx)

        if encoded_pos not in valid_moves:  # Invalid move
            raise ValueError(f"Invalid move for piece {relative_idx} to position {encoded_pos}")

        return True
    
    def update(self, action: tuple, player_idx: int):
        """
        Uses a validated action and updates the game board state
        """
        offset_idx = player_idx * 6 ## Either 0 or 6
        idx, pos = action
        self.game_state.update(offset_idx + idx, pos)


if __name__ == '__main__':
    def test_is_valid():
        print("Initial State")
        board = BoardState()
        assert board.is_valid()

        ## Out of bounds test
        board.update(0,-1)
        print(f"Move (0,-1)")
        assert not board.is_valid()
        
        board.update(0,0)
        print(f"Move (0,0)")
        assert board.is_valid()
        
        ## Out of bounds test
        board.update(0,-1)
        board.update(6,56)
        print(f"Move (0,-1) (6,56)")
        assert not board.is_valid()
        
        ## Overlap test
        board.update(0,0)
        board.update(6,0)
        print(f"Move (0,0) (6,0)")
        assert not board.is_valid()

        ## Ball is on index 0
        board.update(5,1)
        board.update(0,1)
        board.update(6,50)
        print(f"Move (5,1) (0,1) (6,50)")
        assert board.is_valid()

        ## Player is not holding the ball
        board.update(5,0)
        print(f"Move (5,0)")
        assert not board.is_valid()
        
        board.update(5,10)
        print(f"Move (5,10)")
        assert not board.is_valid()

    test_is_valid()