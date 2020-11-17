_11 = 1 << 11        # 32-bit integer
_33 = 1 << 33        # 64-bit integer
_ml = (1 << 63) - 1  # maxlong
_66 = 1 << 66        # BigInteger

def bench_add_32(b):
    for _ in range(b.n):
        (_11 + _11, _11 + _11, _11 + _11, _11 + _11, _11 + _11,
            _11 + _11, _11 + _11, _11 + _11, _11 + _11, _11 + _11)

def bench_add_64(b):
    for _ in range(b.n):
        (_33 + _33, _33 + _33, _33 + _33, _33 + _33, _33 + _33,
            _33 + _33, _33 + _33, _33 + _33, _33 + _33, _33 + _33)

# 64 bit addition overflows
def bench_add_64_overfl(b):
    for _ in range(b.n):
        (_ml + _ml, _ml + _ml, _ml + _ml, _ml + _ml, _ml + _ml,
            _ml + _ml, _ml + _ml, _ml + _ml, _ml + _ml, _ml + _ml)

def bench_add_bigint(b):
    for _ in range(b.n):
        (_66 + _66, _66 + _66, _66 + _66, _66 + _66, _66 + _66,
            _66 + _66, _66 + _66, _66 + _66, _66 + _66, _66 + _66)
