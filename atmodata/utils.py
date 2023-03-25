import time

import numpy as np


def benchmark(dataset, process_time=0, log=True, print_frequency=100):
    it = iter(dataset)
    times = []
    total_size = 0
    i = 0
    while True:
        ts = time.time()
        batch = next(it, None)
        if batch is None:
            break

        times.append(time.time() - ts)
        total_size += batch.nelement() * batch.element_size()
        i += 1

        if process_time:
            time.sleep(process_time / 1000)

        if log and print_frequency and i % print_frequency == 0:
            last_vals = times[-print_frequency:]
            print(f'{1000 * np.mean(last_vals):.0f} ms/batch')

    del it
    times = np.array(times)

    if log:
        print('----------------------------------------------')
        print(f'{np.sum(times):.0f}s total')
        print(f'{times[0]:.1f}s latency')
        print(f'Throughput: {total_size / np.sum(times) / 1e6:.1f} MB/s')
        print(f'Mean: {np.mean(times)*1000:.0f} ms/batch')
        print(f'Std: {np.std(times)*1000:.0f} ms')
        print(f'Median: {np.median(times)*1000:.0f} ms/batch')
        print(f'25th percentile: {np.percentile(times, 25)*1000:.0f} ms/batch')
        print(f'75th percentile: {np.percentile(times, 75)*1000:.0f} ms/batch')
        print(f'99th percentile: {np.percentile(times, 99)*1000:.0f} ms/batch')

    return times
