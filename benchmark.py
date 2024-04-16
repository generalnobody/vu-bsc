# The script containing benchmarking functions and workflows
import time


def benchmark(func, *args, reps):
    result = []
    for _ in range(reps):
        t1 = time.time()
        func(*args)
        t2 = time.time()
        result.append(t2 - t1)
    return result
