_list10k = list(range(10000))
_list10 = list(range(10))
_tuple10k = tuple(range(10000))

def bench_extend(b):
    "Extends an empty list by 10000 items."
    for _ in range(b.n):
        x = []
        x.extend(_list10k)

def bench_append_onebyone(b):
    "Appends 10000 items to an empty list."
    for _ in range(b.n):
        x = []
        for y in _list10k:
            x.append(y)

# This isn't exactly a fair comparison against bench_append_onebyone because warmup effects
# reproducibly favor the test that comes second.
def bench_append_norealloc(b):
    "Appends 10000 items to a list much bigger than that."
    for _ in range(b.n):
        # Exclude list construction from timing. The overhead of stopping and restarting the timer
        # is small (~1%) for 10k items.
        b.stop()
        x = list(_list10k) * 10
        b.start()
        for y in _list10k:
            x.append(y)

def bench_list_list(b):
    "list([10 elements])"
    for _ in range(b.n):
        list(_list10)

def bench_list_in(b):
    "Linear search for last element in list."
    for _ in range(b.n):
        9999 in _list10k

def bench_tuple_in(b):
    "Linear search for last element in tuple."
    for _ in range(b.n):
        9999 in _tuple10k
