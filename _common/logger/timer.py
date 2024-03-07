from datetime import datetime


class Timer:
    def __init__(self, start_now: bool = False):
        self.times = {}
        self.times_from_start = {}
        self.order = []
        self._start = None
        self._end = None
        if start_now:
            self.start_timer()

    def start_timer(self):
        self._start = datetime.now()

    def end_timer(self):
        self._end = datetime.now()

    def time(self, name: str):
        t = datetime.now()
        self.order.append(name)
        self.times[name] = t

    def time_from_start(self, name: str):
        t = datetime.now()
        self.times_from_start[name] = t

    def end_and_get_stats(self, incremental_stats: bool = True) -> dict:
        self.end_timer()
        stats = {}
        old_time = self._start
        for name in self.order:
            new_time = self.times[name]
            reference_time = old_time if incremental_stats else self._start
            stats[name] = (new_time - reference_time).total_seconds()
            old_time = new_time
        for name, end_time in self.times_from_start.items():
            stats[f"start_to_{name}"] = (end_time - self._start).total_seconds()
        stats["total"] = (self._end - self._start).total_seconds()
        return stats
