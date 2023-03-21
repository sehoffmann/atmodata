import time

import numpy as np


def benchmark(dataset, log=True):
    it = iter(dataset)
    times = []

    ts = time.time()
    batch = next(it)
    times.append(time.time() - ts)
    i = 1
    while batch is not None:
        ts = time.time()
        batch = next(it, None)
        times.append(time.time() - ts)
        i += 1
        if i % 250 == 0:
            last_vals = times[-250:]
            print(f'{1000 * sum(last_vals) / len(last_vals):.0f} ms / batch')

    del it
    times = np.array(times)

    if log:
        print(f'{np.sum(times):.0f}s total')
        print(f'{np.mean(times)*1000:.0f} ms / batch')
        print(f'{times[0]:.1f}s latency')

    return times
