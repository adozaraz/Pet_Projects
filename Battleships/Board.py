from Grid import Grid


class Board:
    def __init__(self, grid=Grid(), ships=None):
        if ships is None:
            ships = []
        self.grid = grid
        self.ships = ships

    def acceptGuessFromPlayer(self, coord):
        if self.grid.checkIfShip(coord):
            self.grid.changeType(coord)
            for ship in self.ships:
                ship.update(coord)
        self.checkIfAllShipsSunken()

    def printBoardState(self):
        print(self.grid)

    def checkIfAllShipsSunken(self):
        if sum([int(not i.isSunken()) for i in self.ships]) == 0:
            print('All ships were destroyed')
