from time import time
from datetime import datetime
from collections import deque


class LoopTimer:
    def __init__(self, avg_length=1, update_after=10, target=None):
        self.idx = 0
        self.last_update_idx = 0
        self.avg_length = avg_length

        self.target = target
        self.update_after = update_after
        #self.avg_time = deque(self.avg_length * [float(0)], maxlen=self.avg_length)
        self.avg_time = list()
        self.t = time()
        self.avg_time.append(0.0)
        self.time_samples = self.avg_length * self.update_after

    def update(self, update_text, update_len=1):
        t_now = time()

        self.idx += update_len
        time_past = t_now - self.t
        time_each_iter = time_past / update_len
        for i in range(update_len):
            new_time = time_each_iter
            self.avg_time.append(new_time)

        while len(self.avg_time) > self.avg_length:
            self.avg_time.pop(0)

        if (self.idx - self.last_update_idx) >= self.update_after:
            self.last_update_idx = self.idx
            it_per_second = len(self.avg_time) / sum(self.avg_time)
            if self.target is None:
                ips = "{0:0.2f}".format(it_per_second)
                print(f'\r{update_text}: {self.idx} | {ips} / s     ', end='')
            else:
                ips = "{0: 0.2f}".format(it_per_second)
                time_remaining = (self.target - self.idx) / it_per_second
                days, time_remaining_r = divmod(time_remaining, 60*60*24)
                hours, time_remaining_r = divmod(time_remaining_r, 60*60)
                minutes, seconds = divmod(time_remaining_r, 60)

                time_remaining_s = f"{int(days)}d {int(hours)}h {int(minutes)}min {int(seconds)}sec"

                tr = "{0: 0.2f}".format(time_remaining)
                time_finished = datetime.utcfromtimestamp(t_now + time_remaining + 2*3600).strftime('%Y-%m-%d %H:%M:%S')
                percent = "{0: 0.2f}".format(self.idx / self.target * 100)
                print(f'\r{update_text}: {self.idx} ({percent} % ) | {ips} / s | {time_remaining_s} | {time_finished}        ', end='')
        self.t = time()
        return self.idx
