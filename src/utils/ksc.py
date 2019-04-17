import math
import numpy as np
import random
import pandas as pd

from numpy.linalg import norm


class KSpectralCluster:
    def __init__(self, time_interval, trend_types=['up', 'even', 'down']):
        self.d_range = time_interval
        time_length = len(self.d_range)
        r_scale = 2000
        scale = 0.03
        up_trend = list()
        down_trend = list()
        even_trend = list()
        # upwards trend
        for i in range(time_length):
            assign = scale * (0.5 * math.exp((i - (time_length - 1)) / time_length)
                              + 0.25 + random.randint(0, 100) / r_scale)
            up_trend.append(assign)
        # downwards_trend
        for i in range(time_length):
            assign = scale * (0.5 * (-math.exp((i - (time_length - 1)) / time_length) + 1)
                              + 0.25 + random.randint(0, 100) / r_scale)
            down_trend.append(assign)
        # even_trend
        for i in range(time_length):
            assign = scale * (0.5 + random.randint(0, 100) / r_scale)
            even_trend.append(assign)

        self.trendFrame = pd.DataFrame(index=self.d_range)
        self.trendFrame['up'] = up_trend
        self.trendFrame['down'] = down_trend
        self.trendFrame['even'] = even_trend

        self.trend_types = trend_types

    def assign_cluster(self, df):
        best_dist = float("inf")

        series = df[self.d_range].values.tolist()

        for trend_type in self.trend_types:
            cluster = self.trendFrame[trend_type][self.d_range].values.tolist()
            dist = distance(series, cluster)
            if dist < best_dist:
                best_dist = dist
                optimal_cluster = self.trend_types.index(trend_type)

        return optimal_cluster


def distance(x, y):

    alpha = np.dot(x, y)/math.pow(norm(y), 2)

    return norm(np.subtract(x, np.multiply(y, alpha))) / norm(x)


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


