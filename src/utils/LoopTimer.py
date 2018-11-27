from time import time
from datetime import datetime
from collections import deque


class LoopTimer:
    def __init__(self, avg_length=None, update_after=10, target=None):
        self.idx = 0
        if avg_length is None:
            self.avg_length = 1
        else:
            self.avg_length = avg_length

        self.target = target
        self.update_after = update_after
        self.avg_time = deque(self.avg_length * [float(0)], maxlen=self.avg_length)
        self.t = time()
        self.time_samples = self.avg_length * self.update_after

    def update(self, update_text):
        t_now = time()
        self.avg_time.append(t_now - self.t)

        if self.idx % self.update_after == 0:
            it_per_second = self.avg_length / sum(self.avg_time)
            if self.target is None:
                ips = "{0:0.2f}".format(it_per_second)
                print(f'\r{update_text}: {self.idx} | {ips} / s     ', end='')
            else:
                ips = "{0: 0.2f}".format(it_per_second)
                time_remaining = (self.target - self.idx) / it_per_second
                tr = "{0: 0.2f}".format(time_remaining)
                time_finished = datetime.utcfromtimestamp(t_now + time_remaining + 3600).strftime('%Y-%m-%d %H:%M:%S')
                print(f'\r{update_text}: {self.idx} | {ips} / s | {tr} s | {time_finished}        ', end='')
        self.t = time()
        self.idx += 1

