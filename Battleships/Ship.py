import numpy as np


class Ship:
    def __init__(self, sunken=np.array([]), alive=np.array([])):
        self.sunken_coords = sunken
        self.alive_coords = alive
        self.sunken = False

    def takeHit(self, coords):
        self.alive_coords = np.delete(self.alive_coords, np.argwhere(self.alive_coords == coords))
        np.append(self.sunken_coords, coords)
        if self.alive_coords.size == 0:
            self.sunken = True

    def update(self, coord):
        if not self.sunken:
            if coord in self.alive_coords:
                self.takeHit(coord)

    def isSunken(self):
        return self.sunken
