import time


class PerformanceTimer:
    def __init__(
        self,
        logger,
        start_message: str = "",
        end_message: str = "",
        level: str = "debug",
        run: bool = True,
    ):
        self.to_logger = getattr(logger, level)
        self.start_message = start_message
        self.end_message = end_message
        self.run = run

    def __enter__(self):
        if not self.run:
            return

        if self.start_message:
            self.to_logger(self.start_message)
        self._start = time.perf_counter()

    def __exit__(self):
        if not self.run:
            return

        self._end = time.perf_counter()
        self._elapsed = self._end - self._start
        if self.end_message:
            time_message = f"{self.end_message} ({self._elapsed:.2f} seconds)"
            self.to_logger(time_message)
