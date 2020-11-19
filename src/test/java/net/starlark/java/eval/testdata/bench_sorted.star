def random(seed):
    "Updates seed[0] and returns the next pseudorandom number."
    x = seed[0]
    seed[0] = x + 9319 * 125 % 6011
    return x

def _bench_sort(b, size):
    seed = [0]
    orig = [random(seed) for x in range(size)]
    b.restart()
    for _ in range(b.n):
        copy = orig[:]  # TODO(adonovan): move allocation outside loop
        sorted(copy)

def bench_large(b):
    "Sort an array of a million ints."
    _bench_sort(b, 1000000)

def bench_small(b):
    "Sort an array of ten ints."
    _bench_sort(b, 10)
