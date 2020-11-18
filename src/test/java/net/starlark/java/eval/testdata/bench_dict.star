_d = {x: x for x in range(10)}

def bench_popitem(b):
    for _ in range(b.n):
        d = dict(_d)
        for _j in range(10):
            d.popitem()
