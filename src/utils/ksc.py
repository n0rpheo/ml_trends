import math
import numpy as np

from src.utils.LoopTimer import LoopTimer

class ksc:
    def __init__(self, max_iter = 1000):
        self.max_iter = max_iter
        self.k = 0
        self.n = 0
        self.dim = 0

    def train(self, xs, init_mu):
        self.k = len(init_mu)
        self.n = len(xs)
        self.dim = xs[0].shape[0]

        clusters = dict()

        for j in range(0, self.k):
            clusters[j] = set()

        self.assign_clusters(init_mu, xs, clusters)

        lc = LoopTimer(update_after=1)

        for iter in range(0, self.max_iter):
            old_clusters = dict()

            for j in range(0, self.k):
                old_clusters[j] = clusters[j].copy()

            mu = list()
            for j in range(0, self.k):
                # Calculate M
                M = np.zeros((self.dim, self.dim))
                for i in range(0, self.n):
                    if i in clusters[j]:
                        x_reshape = xs[i].reshape((xs[i].shape[0], 1))

                        matrix = np.subtract(np.ones((self.dim, self.dim)),
                                             np.matmul(x_reshape, x_reshape.T) / math.pow(l2norm(xs[i]), 2))

                        M = np.add(M, matrix)

                w, v = np.linalg.eig(M)

                mu.append(v[np.argmin(w)])

                clusters[j] = set()

            self.assign_clusters(mu, xs, clusters)

            break_cond = True
            for j in range(0, self.k):
                sym_diff = old_clusters[j].symmetric_difference(clusters[j])
                if len(sym_diff) > 0:
                    break_cond = False

            lc.update("KSC")

            if break_cond:
                break

        return clusters, mu

    def assign_clusters(self, mu, series, clusters):
        for i in range(0, self.n):
            best_dist = float("inf")

            for j in range(0, self.k):
                dist = distance(series[i], mu[j])
                if dist < best_dist:
                    best_dist = dist
                    optimal_cluster = j

            clusters[optimal_cluster].add(i)












def distance(x, y):

    alpha = np.dot(x, y)/math.pow(l2norm(y, 1), 2)

    distance = l2norm(np.subtract(x, np.multiply(y, alpha))) / l2norm(x)

    return distance


def l2norm(x, tau=1):

    result = 0
    for i in range(0, len(x)):
        result += math.pow(am1(x, i, tau) + ap1(x, i, tau), 2)

    return math.sqrt(result)


def am1(x, i, tau):
    if i == 0:
        x0 = 0
    else:
        x0 = x[i-1]

    x1 = x[i]

    if 0 <= x1*x0:
        return (tau*abs(x1))/2
    else:
        return (tau*math.pow(x1, 2)) / (2*(abs(x0) + abs(x1)))


def ap1(x, i, tau):
    if i == len(x)-1:
        x1 = 0
    else:
        x1 = x[i+1]

    x0 = x[i]

    if 0 <= x1*x0:
        return (tau*abs(x0))/2
    else:
        return (tau*math.pow(x0, 2)) / (2*(abs(x0) + abs(x1)))


