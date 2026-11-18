# Written by Michael Hoffert and Claude Opus 5.5
# October 2026

# decorators

import functools

def verbose_print(func):
    '''
    allow cmd_* functions to have verbose printing
    '''
    @functools.wraps(func)
    def wrapper(args, *a, **kw):
        def vprint(*msg, **print_kw):
            if getattr(args, "verbose", True):
                print(*msg, **print_kw)
        return func(args, *a, vprint=vprint, **kw)
    return wrapper

import itertools
import sys
import threading
import time
from contextlib import contextmanager

@contextmanager
def spinner(message="loading", interval=0.4, enabled=True):
    """Show 'message.', 'message..', 'message...' on repeat while the block runs."""
    if not enabled:
        yield
        return
    if not sys.stderr.isatty():          # log files, SLURM output: print once, no animation
        print(f"{message}...", file=sys.stderr)
        yield
        return
    stop = threading.Event()

    def animate():
        for dots in itertools.cycle([".", "..", "..."]):
            if stop.is_set():
                break
            sys.stderr.write(f"\r{message}{dots:<3}")
            sys.stderr.flush()
            time.sleep(interval)

    t = threading.Thread(target=animate, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join()
        sys.stderr.write(f"\r{message}... done.\n")