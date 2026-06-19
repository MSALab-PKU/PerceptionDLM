import time
from contextlib import contextmanager
from collections import OrderedDict


class Timer:
    smooth = 0.67

    def __init__(self):
        self.reset()

    def reset(self):
        self.clock_box = OrderedDict()

    def start(self, name: str, skip_first: bool = False):
        if name not in self.clock_box:
            self.clock_box[name] = {}
            if skip_first:
                return
        self.clock_box[name]['start'] = time.time()

    def pause(self, name: str) -> float:
        end = time.time()
        clock = self.clock_box.get(name, {})
        start = clock.pop('start', None)
        if start is None:
            return -1
        pause = end - start + clock.pop('pause', 0.)
        clock['pause'] = pause

        return pause

    def stop(self, name: str) -> float:
        end = time.time()
        clock = self.clock_box.get(name, {})
        start = clock.pop('start', None)
        if start is None:
            if 'pause' in clock:
                start = end
            else:
                return -1
        duration = end - start + clock.pop('pause', 0.)
        if 'duration' in clock:
            clock['duration'] = self.smooth * clock['duration'] + (1 - self.smooth) * duration
        else:
            clock['duration'] = duration

        return duration

    @contextmanager
    def timing(self, name: str, skip_first: bool = False):
        self.start(name, skip_first)
        yield
        self.stop(name)

    def __repr__(self):
        s = []
        for k, v in self.clock_box.items():
            d = v.get('duration', None)
            if d is None:
                continue
            s.append(f"{k}: {d:.4f}s")
        s = "|".join(s)

        return s