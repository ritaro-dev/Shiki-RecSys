import time
from threading import Lock


class ShikimoriRateLimiter:
    """
    Выдерживает минимальный интервал между запросами
    к Shikimori API.
    """

    def __init__(
        self,
        min_interval_seconds: float,
    ) -> None:
        if min_interval_seconds <= 0:
            raise ValueError("min_interval_seconds должен быть больше 0.")

        self.min_interval_seconds = min_interval_seconds
        self._last_request_time: float | None = None
        self._lock = Lock()

    def wait(self) -> None:
        """
        При необходимости ожидает разрешённого времени
        следующего запроса.
        """

        with self._lock:
            now = time.monotonic()

            if self._last_request_time is not None:
                elapsed = now - self._last_request_time
                remaining = self.min_interval_seconds - elapsed

                if remaining > 0:
                    time.sleep(remaining)

            self._last_request_time = time.monotonic()
