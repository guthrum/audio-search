import os
import time

debug = os.environ.get('DEBUG', 'true').lower() == 'true'


def timing(function):
    def new_function(*args, **kwargs):
        start = time.monotonic_ns()
        res = function(*args, **kwargs)
        end = time.monotonic_ns()
        if debug:
            print(f"function {function.__name__} took {(end - start) / 1000000}ms")
        return res

    return new_function
