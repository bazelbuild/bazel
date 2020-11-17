_list1000 = list(range(1000))

def bench_extend(b):
    "Extends an empty list by 1000 items."
    for _ in range(b.n):
        x = []
        x.extend(_list1000)

def bench_append(b):
    "Appends 1000 items to an empty list."
    for _ in range(b.n):
        x = []
        for y in _list1000:
            x.append(y)
