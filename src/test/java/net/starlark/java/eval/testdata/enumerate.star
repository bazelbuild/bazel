# enumerate

assert_eq(enumerate([24, 21, 84]), [(0, 24), (1, 21), (2, 84)])
assert_eq(enumerate([]), [])
assert_eq(enumerate(["only"]), [(0, "only")])

assert_eq(enumerate(list = [10, 20]), [(0, 10), (1, 20)])
assert_eq(enumerate([10, 20], 0), [(0, 10), (1, 20)])
assert_eq(enumerate([10, 20], start = 0), [(0, 10), (1, 20)])

assert_eq(enumerate(["a", "b", "c"], 1), [(1, "a"), (2, "b"), (3, "c")])
assert_eq(enumerate(["a", "b"], start = 100), [(100, "a"), (101, "b")])
assert_eq(enumerate(["a", "b"], -1), [(-1, "a"), (0, "b")])
assert_eq(enumerate([], 42), [])

big = (1 << 31) - 2
assert_eq(enumerate(["x", "y"], big), [(big, "x"), (big + 1, "y")])
assert_fails(
    lambda: enumerate([1], 1 << 40),
    "got 1099511627776 for start, want value in signed 32-bit range",
)

assert_eq(enumerate(()), [])
assert_eq(enumerate((10, 20, 30)), [(0, 10), (1, 20), (2, 30)])
assert_eq(enumerate(range(3)), [(0, 0), (1, 1), (2, 2)])
assert_eq(enumerate("abc".elems()), [(0, "a"), (1, "b"), (2, "c")])
assert_eq(enumerate({"a": 1, "b": 2}), [(0, "a"), (1, "b")])
assert_eq(enumerate({}), [])

result = enumerate([1, 2])
assert_eq(type(result), "list")
assert_eq(type(result[0]), "tuple")
result.append((2, 99))
assert_eq(result, [(0, 1), (1, 2), (2, 99)])

assert_fails(lambda: enumerate("abc"), "type 'string' is not iterable")
assert_fails(lambda: enumerate(1), "type 'int' is not iterable")
assert_fails(lambda: enumerate(None), "type 'NoneType' is not iterable")
assert_fails(lambda: enumerate([1], "x"), "got value of type 'string', want 'int'")
assert_fails(lambda: enumerate([1], 1.5), "got value of type 'float', want 'int'")
assert_fails(lambda: enumerate(), "missing 1 required positional argument: list")
