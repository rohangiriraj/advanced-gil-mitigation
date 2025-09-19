import time
from threading import Thread

COUNT = 100_000_000

def countdown():
    """A simple, CPU-intensive countdown function."""
    n = COUNT
    while n > 0:
        n -= 1

# --- Sequential Execution ---
start_time = time.time()
countdown()
countdown()
end_time = time.time()
print(f"Sequential execution took: {end_time - start_time:.4f} seconds")

# --- Threaded Execution ---
thread1 = Thread(target=countdown)
thread2 = Thread(target=countdown)

start_time = time.time()
thread1.start()
thread2.start()
thread1.join()
thread2.join()
end_time = time.time()
print(f"Threaded execution took:   {end_time - start_time:.4f} seconds")
