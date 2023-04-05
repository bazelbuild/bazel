_haystack = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
_needle = "nostraaad"

def _bench_find_with_start(b, start):
    for _ in range(b.n):
        _haystack.find(_needle, start)

def bench_find(b):
    _bench_find_with_start(b, 0)

def bench_find_with_start(b):
    _bench_find_with_start(b, 1)
