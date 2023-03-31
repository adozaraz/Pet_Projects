import numpy as np
import math

class DifferentiationScheme:
    def __init__(self, params):
        self.l = params["l"]
        self.s = params["s"]
        self.const_alpha = params["alpha"]
        self.const_k = params["k"]
        self.c = params["c"]
        self.T = params["T"]
        self.x = params["x"]
        self.t = params["t"]
        self.I = int(params["I"])
        self.K = int(params["K"])
        self.const_u0 = params["U_0"]
        self.hx = self.l / self.I
        self.ht = self.T / self.K
        self.gamma = 4 * self.const_alpha / np.sqrt(self.s)

    @staticmethod
    def phi(x, l):
        return np.sin(np.pi * x / l) ** 4

    def setNewParams(self, params):
        self.l = params["l"]
        self.s = params["s"]
        self.const_alpha = params["alpha"]
        self.const_k = params["k"]
        self.c = params["c"]
        self.T = params["T"]
        self.x = params["x"]
        self.t = params["t"]
        self.I = int(params["I"])
        self.K = int(params["K"])
        self.const_u0 = params["U_0"]
        self.hx = self.l / self.I
        self.ht = self.T / self.K
        self.gamma = 4 * self.const_alpha / np.sqrt(self.s)

    def SimpleApparentX(self):
        iteration = int(self.t / self.ht)
        x = np.linspace(0, self.l, self.I + 1)
        A_low = np.zeros(self.I + 1)
        gamma = 4 * self.const_alpha / np.sqrt(self.s)
        for i in range(0, self.I + 1):
            A_low[i] = self.const_u0
        for k in range(1, iteration):
            A_top = np.zeros(self.I + 1)
            for i in range(1, self.I):
                temp = self.const_k * (A_low[i + 1] - 2 * A_low[i] + A_low[i - 1]) / self.hx ** 2 - gamma * (
                            A_low[i] - self.const_u0) + self.phi(
                    i * self.hx, self.l)
                A_top[i] = self.ht * temp / self.c + A_low[i]
            A_top[0] = A_top[1] / ((self.const_alpha * self.hx / self.const_k) + 1)
            A_top[self.I] = A_top[self.I - 1] / ((self.const_alpha * self.hx / self.const_k) + 1)
            A_low = A_top

        label = f'ПЯ (T={self.T}, t={self.t}, I={self.I}, K={self.K})'
        return x, A_low, label

    def ModifiedApparentX(self):
        maxNode = int(self.t / self.ht)
        x = np.linspace(0, self.l, self.I + 1)
        lowerLayer = np.zeros(self.I + 1)
        for i in range(lowerLayer.shape[0]):
            lowerLayer[i] = self.const_u0

        for k in range(1, maxNode):
            upperLayer = np.zeros(self.I + 1)
            for i in range(1, self.I):
                node = self.const_k * (lowerLayer[i + 1] - 2 * lowerLayer[i] + lowerLayer[i - 1]) / self.hx ** 2 - \
                       self.gamma * lowerLayer[i] + self.phi(i * self.hx, self.l)
                upperLayer[i] = self.ht * node / self.c + lowerLayer[i]

            node0 = 2 * self.const_k * (lowerLayer[1] - lowerLayer[0] - self.const_alpha
                                        * self.hx * (lowerLayer[0] - self.const_u0) / self.const_k) / self.hx ** 2 - self.gamma * lowerLayer[0] + self.const_u0 + self.phi(0, self.l)
            upperLayer[0] = self.ht * node0 / self.c + lowerLayer[0]
            nodeI = 2 * self.const_k * (lowerLayer[self.I - 1] - lowerLayer[self.I] - self.const_alpha
                                        * self.hx * (lowerLayer[self.I] - self.const_u0) / self.const_k) / self.hx ** 2 - self.gamma * \
                    lowerLayer[self.I] + self.const_u0 + self.phi(self.I * self.hx, self.l)
            upperLayer[self.I] = self.ht * nodeI / self.c + lowerLayer[self.I]
            lowerLayer = upperLayer

        label = f'МЯ (T={self.T}, t={self.t}, I={self.I}, K={self.K})'
        return x, lowerLayer, label

    def SimpleImplicitX(self):
        pass

    def ModifiedImplicitX(self):
        x = np.linspace(0, 1, self.I + 1)  ########
        maxNode = int(self.t / self.ht)
        betta = self.const_alpha / self.const_k
        mu = self.const_k / self.c
        gamma = 4 * self.const_alpha / (self.s ** 0.5)
        a = 1 + 2 * mu * self.ht / (self.hx ** 2) + gamma * self.ht / self.c
        b = mu * self.ht / (self.hx ** 2)
        g_i = np.zeros(self.I + 1)
        phi = np.zeros(self.I + 1)
        for i in range(0, self.I):
            phi[i] = math.sin(math.pi * x[i] / self.l) ** 4
            g_i[i] = self.ht / self.c * (gamma * self.const_u0 + phi[i])
        U = np.zeros((self.K, self.I + 1))
        U[0, 0:self.I] = self.const_u0  # u_i_0 = u0
        g_shtrih_i = np.zeros((self.K + 1, self.I + 1))
        v1 = np.zeros(self.K + 1)
        v2 = np.zeros(self.K + 1)
        alpha_i = np.zeros(self.I + 1)
        betta_i = np.zeros((self.K, self.I + 1))
        d1 = d2 = 1 + 2 * mu * self.ht / (self.hx ** 2) + gamma * self.ht / self.c + 2 * betta * mu * self.ht / self.hx
        w1 = w2 = 2 * mu * self.ht / (self.hx ** 2)
        alpha_i[1] = b / (a - b * w1 / d1)
        for k in range(1, self.K):
            for i in range(0, self.I):
                g_shtrih_i[k - 1, i] = U[k - 1, i] + g_i[i]

            v1[k - 1] = U[k - 1, 0] + self.const_u0 * (2 * betta * mu * self.ht / self.hx +
                                                       gamma * self.ht / self.c) + self.ht / self.c * phi[0]
            betta_i[k, 1] = (g_shtrih_i[k - 1, 1] + b * v1[k - 1] / d1) / (a - b * w1 / d1)

            for i in range(2, self.I - 1):
                alpha_i[i] = b / (a - b * alpha_i[i - 1])
                betta_i[k, i] = (g_shtrih_i[k - 1, i] + b * betta_i[k, i - 1]) / (a - b * alpha_i[i - 1])
            v2[k - 1] = U[k - 1, self.I] + self.const_u0 * (2 * betta * mu * self.ht / self.hx +
                                                            gamma * self.ht / self.c) + self.ht / self.c * phi[
                            self.I - 1]
            U[k, self.I] = (v2[k - 1] + w2 * betta_i[k, self.I - 1]) / (d2 - w2 * alpha_i[self.I - 1])
            for i in range(self.I - 1, 1, -1):
                U[k, i] = alpha_i[i] * U[k, i + 1] + betta_i[k, i]
            U[k, 0] = w1 / d1 * U[k, 1] + v1[k - 1] / d1
        label = f'МН (T={self.T}, t={self.t}, I={self.I}, K={self.K})'
        return x, U[maxNode], label

    def SimpleApparentT(self):
        gamma = 4 * self.const_alpha / np.sqrt(self.s)
        iteration = int(self.x / self.hx)
        t = np.linspace(0, self.T, self.K + 1)
        A_low = np.zeros(self.I + 1)
        for i in range(0, self.I + 1):
            A_low[i] = self.const_u0
        U = [A_low[iteration]]
        for k in range(1, self.K + 1):
            A_top = np.zeros(self.I + 1)
            for i in range(1, self.I):
                temp = self.const_k * (A_low[i + 1] - 2 * A_low[i] + A_low[i - 1]) / self.hx ** 2 - gamma * (A_low[i] - self.const_u0) + self.phi(
                    i * self.hx, self.l)
                A_top[i] = self.ht * temp / self.c + A_low[i]

            A_top[0] = A_top[1] / ((self.const_alpha * self.hx / self.const_k) + 1)
            A_top[self.I] = A_top[self.I - 1] / ((self.const_alpha * self.hx / self.const_k) + 1)
            A_low = A_top

            U.append(A_low[iteration])

        label = f'ПЯ (x={self.x}, I={self.I}, K={self.K})'
        return t, U, label

    def ModifiedApparentT(self):
        maxNode = int(self.x / self.hx)
        t = np.linspace(0, self.T, self.K + 1)
        lowerLayer = np.zeros(self.I + 1)

        for i in range(lowerLayer.shape[0]):
            lowerLayer[i] = self.const_u0

        U = [lowerLayer[maxNode]]

        for k in range(1, self.K + 1):
            upperLayer = np.zeros(self.I + 1)
            for i in range(1, self.I):
                node = self.const_k * (lowerLayer[i + 1] - 2 * lowerLayer[i] + lowerLayer[i - 1]) / self.hx ** 2 - \
                       self.gamma * lowerLayer[i] + self.phi(i * self.hx, self.l)
                upperLayer[i] = self.ht * node / self.c + lowerLayer[i]

            node0 = 2 * self.const_k * (lowerLayer[1] - lowerLayer[0] - self.const_alpha
                                        * self.hx * (lowerLayer[
                                                         0] - self.const_u0) / self.const_k) / self.hx ** 2 - self.gamma * \
                    lowerLayer[0] + self.const_u0 + self.phi(0, self.l)
            upperLayer[0] = self.ht * node0 / self.c + lowerLayer[0]
            nodeI = 2 * self.const_k * (lowerLayer[self.I - 1] - lowerLayer[self.I] - self.const_alpha
                                        * self.hx * (lowerLayer[
                                                         self.I] - self.const_u0) / self.const_k) / self.hx ** 2 - self.gamma * \
                    lowerLayer[self.I] + self.const_u0 + self.phi(self.I * self.hx, self.l)
            upperLayer[self.I] = self.ht * nodeI / self.c + lowerLayer[self.I]
            lowerLayer = upperLayer

            U.append(lowerLayer[maxNode])

        label = f'МЯ (x={self.x}, I={self.I}, K={self.K})'

        return t, U, label

    def SimpleImplicitT(self):
        pass

    def ModifiedImplicitT(self):
        t = np.linspace(0, self.T, self.K + 1)
        betta = self.const_alpha / self.const_k
        mu = self.const_k / self.c
        gamma = 4 * self.const_alpha / (self.s ** 0.5)
        a = 1 + 2 * mu * self.ht / (self.hx ** 2) + gamma * self.ht / self.c
        b = mu * self.ht / (self.hx ** 2)
        g_i = np.zeros(self.I + 1)
        phi = np.zeros(self.I + 1)
        for i in range(0, self.I):
            phi[i] = math.sin(math.pi * self.x / self.l) ** 4
            g_i[i] = self.ht / self.c * (gamma * self.const_u0 + phi[i])
        U = np.zeros((self.K + 1, self.I + 1))
        U[0, 0:self.I] = self.const_u0  # u_i_0 = u0
        g_shtrih_i = np.zeros((self.K + 1, self.I + 1))
        v1 = np.zeros(self.K + 1)
        v2 = np.zeros(self.K + 1)
        alpha_i = np.zeros(self.I + 1)
        betta_i = np.zeros((self.K, self.I + 1))
        d1 = d2 = 1 + 2 * mu * self.ht / (self.hx ** 2) + gamma * self.ht / self.c + 2 * betta * mu * self.ht / self.hx
        w1 = w2 = 2 * mu * self.ht / (self.hx ** 2)
        alpha_i[1] = b / (a - b * w1 / d1)
        for k in range(1, self.K):
            for i in range(0, self.I):
                g_shtrih_i[k - 1, i] = U[k - 1, i] + g_i[i]

            v1[k - 1] = U[k - 1, 0] + self.const_u0 * (2 * betta * mu * self.ht / self.hx +
                                                       gamma * self.ht / self.c) + self.ht / self.c * phi[0]
            betta_i[k, 1] = (g_shtrih_i[k - 1, 1] + b * v1[k - 1] / d1) / (a - b * w1 / d1)

            for i in range(2, self.I - 1):
                alpha_i[i] = b / (a - b * alpha_i[i - 1])
                betta_i[k, i] = (g_shtrih_i[k - 1, i] + b * betta_i[k, i - 1]) / (a - b * alpha_i[i - 1])
            v2[k - 1] = U[k - 1, self.I] + self.const_u0 * (2 * betta * mu * self.ht / self.hx +
                                                            gamma * self.ht / self.c) + self.ht / self.c * phi[
                            self.I - 1]
            U[k, self.I] = (v2[k - 1] + w2 * betta_i[k, self.I - 1]) / (d2 - w2 * alpha_i[self.I - 1])
            for i in range(self.I - 1, 1, -1):
                U[k, i] = alpha_i[i] * U[k, i + 1] + betta_i[k, i]
            U[k, 0] = w1 / d1 * U[k, 1] + v1[k - 1] / d1
        label = f'МН (x={self.x}, I={self.I}, K={self.K})'
        return t, U[:, 2], label
