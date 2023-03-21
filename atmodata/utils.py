import time

import numpy as np


def benchmark(dataset, process_time=0, log=True):
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
        if process_time:
            time.sleep(process_time / 1000)
        i += 1
        if log and i % 250 == 0:
            last_vals = times[-250:]
            print(f'{1000 * sum(last_vals) / len(last_vals):.0f} ms/batch')

    del it
    times = np.array(times)

    if log:
        print(f'{np.sum(times):.0f}s total')
        print(f'{times[0]:.1f}s latency')
        print(f'Mean: {np.mean(times)*1000:.0f} ms/batch')
        print(f'Median: {np.median(times)*1000:.0f} ms/batch')
        print(f'25th percentile: {np.percentile(times, 25)*1000:.0f} ms/batch')
        print(f'75th percentile: {np.percentile(times, 75)*1000:.0f} ms/batch')
        print(f'99th percentile: {np.percentile(times, 99)*1000:.0f} ms/batch')

    return times
