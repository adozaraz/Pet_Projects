from itertools import permutations
from statistics import mean


def findMeanMatrix(inputMatrix: list, size: int):
    mainDiag = [i for i in range(size)]
    optDiag = [i for i in range(size - 1, -1, -1)]
    numbers = [inputMatrix[i][j] for i, j in permutations(range(size), 2) if j < mainDiag[i] and j < optDiag[i]]
    return mean(numbers)


size = 4
inputMatrix = [[i*size+j+1 for j in range(size)] for i in range(size)]
print(findMeanMatrix(inputMatrix, size))
