def bench_dot_expr_native(b):
    "Benchmark dot-expression performance for native methods"
    for _ in range(b.n):
        # Use several different object types and and identifiers
        # to make benchmark results more stable.
        "abc".startswith
        [].append
        "cde".endswith
        [].extend
        {}.pop
        "fg".rsplit
        [].pop
        "".lower
        "".lstrip
        [].remove
