from Board import Board
from Grid import Grid
from Ship import Ship
import numpy as np


def main():
    gameState = Board(grid=Grid(np.eye(10)), ships=[Ship(alive=np.array([(i, i)])) for i in range(10)])
    gameState.printBoardState()
    for i in range(10):
        gameState.acceptGuessFromPlayer((i, i))
    gameState.printBoardState()



if __name__ in '__main__':
    main()
