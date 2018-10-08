from time import time
from collections import deque


class LoopTimer:
    def __init__(self, avg_length=None, update_after=10, target=None):
        self.idx = 0
        if avg_length is None:
            self.avg_length = update_after
        else:
            self.avg_length = avg_length

        self.target = target
        self.update_after = update_after
        self.avg_time = deque(self.avg_length * [float(0)], maxlen=self.avg_length)
        self.t = time()
        self.time_samples = self.avg_length * self.update_after

    def update(self, update_text):
        if self.idx % self.update_after == 0:
            self.avg_time.append(time() - self.t)
            it_per_second = self.time_samples / sum(self.avg_time)
            if self.target is None:
                ips = "{0:0.2f}".format(it_per_second)
                print(f'\r{update_text}: {self.idx} | {ips} / s     ', end='')
            else:
                ips = "{0: 0.2f}".format(it_per_second)
                time_remaining = "{0: 0.2f}".format((self.target - self.idx) / it_per_second)
                print(f'\r{update_text}: {self.idx} | {ips} / s | {time_remaining} s     ', end='')
                #print('\r' + update_text + ": " + str(self.idx) + ' | ' + "{0:0.2f}".format(it_per_second) + ' / s     ', end='')
            self.t = time()
        self.idx += 1

