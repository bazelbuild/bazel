def bench_call_type(b):
    for _ in range(b.n):
        type(1)

def bench_call_list_append(b):
    for _ in range(b.n):
        [].append("foo")

def bench_call_dict_get(b):
    d = {"foo": "bar"}
    for _ in range(b.n):
        d.get("baz")

def bench_call_dict_get_none(b):
    d = {"foo": "bar"}
    for _ in range(b.n):
        d.get("baz", None)

def bench_call_bool(b):
    for _ in range(b.n):
        bool()
