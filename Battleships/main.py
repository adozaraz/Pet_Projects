import numpy as np
from NeuralNetwork.BattleshipEnvironment import BattleshipEnvironment


def main():
    print('Creating environment')
    environment = BattleshipEnvironment()
    environment.print_board()


if __name__ in '__main__':
    main()
