_list1000 = list(range(1000))
_list10 = list(range(10))

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

def bench_list_list(b):
    "list([10 elements])"
    for _ in range(b.n):
        list(_list10)
