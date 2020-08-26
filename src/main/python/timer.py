import time

def timed(func):
    print(f"Timing {func}")
    start = time.time()
    func()
    end = time.time()
    print(f"Time: {end-start}")