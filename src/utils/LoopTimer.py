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
        self.avg_time = deque()
        self.t = time()
        self.avg_time.append(0.0)
        self.time_samples = self.avg_length * self.update_after

    def update(self, update_text, update_len=1):
        t_now = time()
        time_past = t_now - self.t
        self.t = time()

        self.idx += update_len
        time_each_iter = time_past / update_len
        for i in range(update_len):
            new_time = time_each_iter
            self.avg_time.append(new_time)

        while len(self.avg_time) > self.avg_length:
            self.avg_time.popleft()

        if (self.idx - self.last_update_idx) >= self.update_after:
            self.last_update_idx = self.idx
            it_per_second = len(self.avg_time) / sum(self.avg_time)
            if it_per_second < 1:
                tr = 1 / it_per_second
                hours, tr = divmod(tr, 60 * 60)
                minutes, seconds = divmod(tr, 60)
                ips = f"{int(hours)}h {int(minutes)}min {int(seconds)}sec per Iteration"
            else:
                ips = f"{round(it_per_second, 2)} / s"
            if self.target is None:
                print(f'\r{update_text}: {self.idx} | {ips}     ', end='')
            else:

                time_remaining = (self.target - self.idx) / it_per_second
                days, time_remaining_r = divmod(time_remaining, 60*60*24)
                hours, time_remaining_r = divmod(time_remaining_r, 60*60)
                minutes, seconds = divmod(time_remaining_r, 60)

                time_remaining_s = f"{int(days)}d {int(hours)}h {int(minutes)}min {int(seconds)}sec"

                time_finished = datetime.utcfromtimestamp(t_now + time_remaining + 2*3600).strftime('%Y-%m-%d %H:%M:%S')
                percent = round(self.idx / self.target * 100, 2)
                print(f'\r{update_text}: {self.idx} ({percent} % ) | {ips} | {time_remaining_s} | {time_finished}        ', end='')
        return self.idx
