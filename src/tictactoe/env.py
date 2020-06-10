from agent import Agent

import numpy as np

class Env():
    
    def __init__(self, agent_1, agent_2):
        '''
        Initialize Env object.
        '''
        self.board = np.zeros(9, dtype=np.intc)
        self.done = False
        self.agent_1 = agent_1
        self.agent_2 = agent_2

    def play_move(self, move, agent_id):
        '''
        Play given move.

        Returns True if game is over after move is
        played and False otherwise. Reward handling
        is abstracted out of this function.
        '''
        if self.board[move] == 0:
            self.board[move] = agent_id
            if self.has_won(agent_id) or self.is_tie():
                self.done = True
            return self.done
        else:
            raise ValueError("Invalid move.")

    def reset(self):
        '''
        Resets parameters of Env.
        '''
        self.board = np.zeros(9, dtype=np.intc)
        self.done = False

    def has_won(self, agent_id):
        '''
        Returns whether the current game is over.
        '''
        return self.same_type(0,1,2,agent_id) or self.same_type(3,4,5,agent_id) or self.same_type(6,7,8,agent_id) \
                or self.same_type(0,3,6,agent_id) or self.same_type(1,4,7,agent_id) or self.same_type(2,5,8,agent_id) \
                or self.same_type(0,4,8,agent_id) or self.same_type(2,4,6,agent_id)

    def same_type(self, first, second, third, agent_id):
        '''
        Returns whether the indices at first, second
        and third are equal to agent_id
        '''
        return (self.board[first] == self.board[second] == self.board[third] == agent_id)

    def is_tie(self):
        '''
        Returns whether there is a tie in the game.
        '''
        return self.is_filled and not self.has_won(self.agent_1.agent_id) and not self.has_won(self.agent_2.agent_id)

    def is_filled(self):
        '''
        Returns whether the current board is filled.
        '''
        for i in np.nditer(self.board):
            if i == 0:
                return False
        return True

    def get_available_moves(self):
        '''
        Returns available moves in board.
        '''
        available_moves = []
        for i in range(9):
            if self.board[i] == 0:
                available_moves.append(i)

        return available_moves

    def render(self):
        '''
        Prints current state of board.
        '''
        res = ''.join([
            str(self.board[i]) + '\n' if (i + 1) % 3 == 0 else str(self.board[i]) for i in range(9)
            ])
        print(res, end = '')

    def get_condensed_state(self):
        '''
        Returns condenses state of board.

        For example, a board with state

            1 0 2
            2 1 0
            1 2 0

        would return 102210120
        '''
        return ''.join([str(i) for i in np.nditer(self.board)])
