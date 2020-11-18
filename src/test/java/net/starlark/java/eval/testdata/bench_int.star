_11 = 1 << 11  # 32-bit integer
_33 = 1 << 33  # 64-bit integer
_maxlong = (1 << 63) - 1
_66 = 1 << 66  # BigInteger

def bench_add32(b):
    for _ in range(b.n):
        _11 + _11
        _11 + _11
        _11 + _11
        _11 + _11
        _11 + _11
        _11 + _11
        _11 + _11
        _11 + _11
        _11 + _11
        _11 + _11

def bench_add64(b):
    for _ in range(b.n):
        _33 + _33
        _33 + _33
        _33 + _33
        _33 + _33
        _33 + _33
        _33 + _33
        _33 + _33
        _33 + _33
        _33 + _33
        _33 + _33

def bench_add64_overflow(b):
    for _ in range(b.n):
        _maxlong + _maxlong
        _maxlong + _maxlong
        _maxlong + _maxlong
        _maxlong + _maxlong
        _maxlong + _maxlong
        _maxlong + _maxlong
        _maxlong + _maxlong
        _maxlong + _maxlong
        _maxlong + _maxlong
        _maxlong + _maxlong

def bench_addbig(b):
    for _ in range(b.n):
        _66 + _66
        _66 + _66
        _66 + _66
        _66 + _66
        _66 + _66
        _66 + _66
        _66 + _66
        _66 + _66
        _66 + _66
        _66 + _66
