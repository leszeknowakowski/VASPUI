import contextlib
import functools
import os
import time


TIMING_ENV_VAR = "VASPUI_PROFILE_TIMING"


def timing_enabled():
    return os.environ.get(TIMING_ENV_VAR, "").lower() in {"1", "true", "yes", "on"}


def log_timing(name, elapsed, **details):
    if not timing_enabled():
        return
    detail_text = " ".join(f"{key}={value}" for key, value in details.items() if value is not None)
    if detail_text:
        detail_text = " " + detail_text
    print(f"[timing] {name}: {elapsed * 1000:.2f} ms{detail_text}")


@contextlib.contextmanager
def timed_block(name, **details):
    start = time.perf_counter()
    try:
        yield
    finally:
        log_timing(name, time.perf_counter() - start, **details)


def timed(name=None):
    def decorator(func):
        timer_name = name or f"{func.__module__}.{func.__qualname__}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                log_timing(timer_name, time.perf_counter() - start)

        return wrapper

    return decorator
