# The script containing benchmarking functions and workflows
import time


def benchmark(func, *args, reps=0):
    if reps <= 0:
        print("error: wrong benchmark reps value")
        return None

    result = []
    # Repeat benchmark 'reps' times
    for _ in range(reps):
        t1 = time.time()
        func(*args)
        t2 = time.time()
        result.append(t2 - t1)
    return result
