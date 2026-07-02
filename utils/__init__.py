import sys
import os
from contextlib import contextmanager

@contextmanager
def suppress_stdout():
    """A context manager to suppress stdout temporarily."""
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout
