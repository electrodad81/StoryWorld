import time, random

def retry_call(fn, *, tries=3, base=0.5, factor=2.0, max_delay=8.0, retriable=(Exception,)):
    """Simple exponential backoff with jitter for non-streaming calls."""
    delay = base
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except retriable:
            if attempt == tries:
                raise
            time.sleep(delay + random.uniform(0, delay / 4))
            delay = min(max_delay, delay * factor)