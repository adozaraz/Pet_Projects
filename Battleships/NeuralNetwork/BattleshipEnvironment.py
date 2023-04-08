from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types


# Стандартный размнер поля, только квадратное
BOARD_SIZE = 10

MAX_STEPS_PER_EPISODE = BOARD_SIZE**2

# Награды
HIT_REWARD = 1
MISS_REWARD = 0
REPEAT_STRIKE_REWARD = -1

FINISHED_GAME_REWARD = 10
UNFINISHED_GAME_REWARD = -10

HIDDEN_OCCUPIED = 1
HIDDEN_UNOCCUPIED = 0

VISIBLE_BOARD_CELL_HIT = 1
VISIBLE_BOARD_CELL_MISS = -1
VISIBLE_BOARD_CELL_UNTRIED = 0


def generateRandomBattleshipBoard(size):
    pass


class BattleshipEnvironment(py_environment.PyEnvironment):
    def __init__(self,
                 boardSize = BOARD_SIZE,
                 discount = 0.9,
                 maxSteps = MAX_STEPS_PER_EPISODE):
        super().__init__()
        self._strike_count = None
        self._hidden_board = None
        self._visible_board = None
        self._battleships_count = None
        self._hit_count = None
        self._boardSize = boardSize
        self._discount = discount
        self._maxSteps = maxSteps
        self._observation_spec = array_spec.BoundedArraySpec(
            (self._boardSize, self._boardSize),
            np.float32,
            minimum=VISIBLE_BOARD_CELL_MISS,
            maximum=VISIBLE_BOARD_CELL_HIT
        )
        self._action_spec = array_spec.BoundedArraySpec((), np.int32, minimum=0, maximum=self._boardSize**2-1)
        self._current_time_step = ts.time_step_spec(self._observation_spec)
        self._episode_ended = False
        self.setUpBoards()

    def setUpBoards(self):
        self._strike_count = 0
        self._hit_count = 0
        self._battleships_count = 20
        self._visible_board = np.zeros((self._boardSize, self._boardSize))
        self._hidden_board = generateRandomBattleshipBoard((self._boardSize, self._boardSize))

    def observation_spec(self) -> types.NestedArraySpec:
        return self._observation_spec

    def action_spec(self) -> types.NestedArraySpec:
        return self._action_spec

    def current_time_step(self) -> ts.TimeStep:
        return self._current_time_step

    def _step(self, action: types.NestedArray) -> ts.TimeStep:
        pass

    def _reset(self) -> ts.TimeStep:
        self._episode_ended = False
        self.setUpBoards()
        return ts.restart(np.array(self._visible_board, dtype=np.float32))
