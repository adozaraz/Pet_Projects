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

SHIPS_COUNT = 20

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

#Direction Generation
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


def generateRandomBattleshipBoard(size):
    board = np.zeros(size)
    i = 0
    # size 4 ship
    while i < 1:
        if randomizeShipPlace(board, 4):
            i += 1
    i = 0
    # Size 3 ship
    while i < 2:
        if randomizeShipPlace(board, 3):
            i += 1

    i = 0
    # Size 2 ship
    while i < 3:
        if randomizeShipPlace(board, 2):
            i += 1

    i = 0
    # Size 1 ship
    while i < 4:
        if randomizeShipPlace(board, 1):
            i += 1

    return board


def checkIfNeighbours(board, coord):
    upperLeftX = coord[0] - 1
    upperLeftY = coord[1] - 1
    lowerRightX = coord[0] + 2
    lowerRightY = coord[1] + 2
    if np.count_nonzero(board[upperLeftX:lowerRightX, upperLeftY:lowerRightY]) != 0:
        return True
    return False


def randomizeShipPlace(board, shipSize):
    cell = np.random.randint(0, 10, size=2)
    if board[cell[0], cell[1]] == HIDDEN_UNOCCUPIED:
        direction = np.random.randint(0, 4)
        if checkIfPlacementIsPossible(board, cell, shipSize, direction):
            setShip(board, cell, direction, shipSize)
            return True
    return False


def checkIfPlacementIsPossible(board, coords, shipSize, direction):
    if direction == UP:
        if coords[1] + shipSize > BOARD_SIZE:
            return False
        for y in range(coords[1], coords[1] + shipSize):
            if board[coords[0], y] != HIDDEN_UNOCCUPIED or checkIfNeighbours(board, (coords[0], y)):
                return False
        return True
    elif direction == RIGHT:
        if coords[0] + shipSize > BOARD_SIZE:
            return False
        for x in range(coords[0], coords[0] + shipSize):
            if board[x, coords[1]] != HIDDEN_UNOCCUPIED or checkIfNeighbours(board, (x, coords[1])):
                return False
        return True
    elif direction == DOWN:
        if coords[1] - shipSize > BOARD_SIZE:
            return False
        for y in range(coords[1], coords[1] - shipSize, -1):
            if board[coords[0], y] != HIDDEN_UNOCCUPIED or checkIfNeighbours(board, (coords[0], y)):
                return False
        return True
    elif direction == LEFT:
        if coords[0] - shipSize > BOARD_SIZE:
            return False
        for x in range(coords[0], coords[0] - shipSize, -1):
            if board[x, coords[1]] != HIDDEN_UNOCCUPIED or checkIfNeighbours(board, (x, coords[1])):
                return False
        return True


def setShip(board, startingCoord, direction, size):
    if direction == UP:
        for y in range(startingCoord[1], startingCoord[1]+size):
            board[startingCoord[0], y] = HIDDEN_OCCUPIED
    elif direction == RIGHT:
        for x in range(startingCoord[0], startingCoord[0] + size):
            board[x, startingCoord[1]] = HIDDEN_OCCUPIED
    elif direction == DOWN:
        for y in range(startingCoord[1], startingCoord[1] - size, -1):
            board[startingCoord[0], y] = HIDDEN_OCCUPIED
    elif direction == LEFT:
        for x in range(startingCoord[0], startingCoord[0] - size, -1):
            board[x, startingCoord[1]] = HIDDEN_OCCUPIED


class BattleshipEnvironment(py_environment.PyEnvironment):
    def __init__(self,
                 boardSize = BOARD_SIZE,
                 discount = 0.9,
                 maxSteps = MAX_STEPS_PER_EPISODE):
        super().__init__()
        # HIT COUNTS
        self._strike_count = None # Количество выстрелов
        self._hit_count = None # Количество попаданий
        # BOARDS
        self._hidden_board = None
        self._visible_board = None
        self._boardSize = boardSize
        # INFORMATION ABOUT SIMULATION
        self._battleships_count = None
        self._discount = discount
        self._maxSteps = maxSteps
        # NN SPECIFIC DATA
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

    def print_board(self):
        for i in self._hidden_board:
            print(''.join(['.' if t == HIDDEN_UNOCCUPIED else 'X' for t in i]))

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
        if self._hit_count == SHIPS_COUNT:
            self._episode_ended = True
            return self.reset()

        if self._strike_count == MAX_STEPS_PER_EPISODE:
            self.reset()
            return ts.termination(np.array(self._visible_board, dtype=np.float32),
                                  UNFINISHED_GAME_REWARD)

        self._strike_count += 1
        action_x = action // self._boardSize
        action_y = action % self._boardSize
        # Hit
        if self._hidden_board[action_x, action_y] == HIDDEN_OCCUPIED:
            # Unrepeated move
            if self._visible_board[action_x, action_y] == VISIBLE_BOARD_CELL_UNTRIED:
                self._hit_count += 1
                # Successful hit
                self._visible_board[action_x, action_y] = VISIBLE_BOARD_CELL_HIT
                if self._hit_count == SHIPS_COUNT:
                    self._episode_ended = True
                    return ts.termination(np.array(self._visible_board, dtype=np.float32),
                                          FINISHED_GAME_REWARD)
                else:
                    self._episode_ended = False
                    return ts.transition(np.array(self._visible_board, dtype=np.float32),
                                         HIT_REWARD, self._discount)
            # Repeated move
            else:
                self._episode_ended = False
                return ts.transition(np.array(self._visible_board, dtype=np.float32),
                                     REPEAT_STRIKE_REWARD, self._discount)
        # Miss
        else:
            self._episode_ended = False
            self._visible_board[action_x, action_y] = VISIBLE_BOARD_CELL_MISS
            return ts.transition(np.array(self._visible_board, dtype=np.float32),
                                 MISS_REWARD, self._discount)

    def _reset(self) -> ts.TimeStep:
        self._episode_ended = False
        self.setUpBoards()
        return ts.restart(np.array(self._visible_board, dtype=np.float32))
