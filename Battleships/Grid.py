import numpy as np
from BoardStates import BoardStates


class Grid:
    def __init__(self, grid=np.zeros((10, 10))):
        # Grid
        # 0 - Empty space, 1 - ship/part of ship, 2 - destroyed ship
        self.grid = grid

    def __str__(self):
        res = ''
        for row in self.grid:
            for column in row:
                if column == BoardStates.EMPTY.value:
                    res += '.'
                elif column == BoardStates.SHIP.value:
                    res += '@'
                elif column == BoardStates.SUNKEN.value:
                    res += 'X'
            res += '\n'
        return res

    def checkIfShip(self, coord):
        if self.grid[coord] == BoardStates.SHIP.value:
            return True
        return False

    def changeType(self, coord):
        self.grid[coord] = BoardStates.SUNKEN.value
