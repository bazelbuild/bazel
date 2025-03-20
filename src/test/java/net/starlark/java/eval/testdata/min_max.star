# min / max

# min
assert_eq(min("abcdefxyz".elems()), "a")
assert_eq(min("test", "xyz"), "test")
assert_eq(min([4, 5], [1]), [1])
assert_eq(min([1, 2], [3]), [1, 2])
assert_eq(min([1, 5], [1, 6], [2, 4], [0, 6]), [0, 6])
assert_eq(min([-1]), -1)
assert_eq(min([5, 2, 3]), 2)
assert_eq(min({1: 2, -1: 3}), -1)  # a single dict argument is treated as its sequence of keys
assert_eq(min({2: None}), 2)  # a single dict argument is treated as its sequence of keys
assert_eq(min(-1, 2), -1)
assert_eq(min(5, 2, 3), 2)
assert_eq(min(1, 1, 1, 1, 1, 1), 1)
assert_eq(min([1, 1, 1, 1, 1, 1]), 1)
assert_eq(min([None]), None)  # not an error: no comparisons required
assert_fails(lambda: min(1), "type 'int' is not iterable")
assert_fails(lambda: min(), "expected at least one item")
assert_fails(lambda: min([]), "expected at least one item")
assert_fails(lambda: min(1, "2", True), "unsupported comparison: int <=> string")
assert_fails(lambda: min([1, "2", True]), "unsupported comparison: int <=> string")

# min with key
assert_eq(min("aBcDeFXyZ".elems(), key = lambda s: s.upper()), "a")
assert_eq(min("test", "xyz", key = len), "xyz")
assert_eq(min([4, 5], [1], key = lambda x: x), [1])
assert_eq(min([1, 2], [3], key = lambda x: x), [1, 2])
assert_eq(min([1, 5], [1, 6], [2, 4], [0, 6], key = lambda x: x), [0, 6])
assert_eq(min([1, 5], [1, 6], [2, 4], [0, 6], key = lambda x: x[1]), [2, 4])
assert_eq(min([-1], key = lambda x: x), -1)
assert_eq(min([5, 2, 3], key = lambda x: x), 2)
assert_eq(min({1: 2, -1: 3}, key = lambda x: x), -1)  # a single dict argument is treated as its sequence of keys
assert_eq(min({2: None}, key = lambda x: x), 2)  # a single dict argument is treated as its sequence of keys
assert_eq(min(-1, 2, key = lambda x: x), -1)
assert_eq(min(5, 2, 3, key = lambda x: x), 2)
assert_eq(min(1, 1, 1, 1, 1, 1, key = lambda x: -x), 1)
assert_eq(min([1, 1, 1, 1, 1, 1], key = lambda x: -x), 1)
assert_fails(lambda: min(1, key = lambda x: x), "type 'int' is not iterable")
assert_fails(lambda: min(key = lambda x: x), "expected at least one item")
assert_fails(lambda: min([], key = lambda x: x), "expected at least one item")
assert_fails(lambda: min([1], ["2"], [True], key = lambda x: x[0]), "unsupported comparison: (int <=> string|string <=> int)")
assert_fails(lambda: min([[1], ["2"], [True]], key = lambda x: x[0]), "unsupported comparison: (int <=> string|string <=> int)")

# verify min with key chooses first value with minimal key
assert_eq(min(1, -1, -2, 2, key = abs), 1)
assert_eq(min([1, -1, -2, 2], key = abs), 1)

# min with failing key
assert_fails(lambda: min(0, 1, 2, 3, 4, key = lambda x: "foo".elems()[x]), "index out of range \\(index is 3, but sequence has 3 elements\\)")
assert_fails(lambda: min([0, 1, 2, 3, 4], key = lambda x: "foo".elems()[x]), "index out of range \\(index is 3, but sequence has 3 elements\\)")

# min with non-callable key
assert_fails(lambda: min(1, 2, 3, key = "hello"), "parameter 'key' got value of type 'string', want 'callable or NoneType'")
assert_fails(lambda: min([1, 2, 3], key = "hello"), "parameter 'key' got value of type 'string', want 'callable or NoneType'")

# verify min with key invokes key callback exactly once per item
def make_counting_identity():
    call_count = {}

    def counting_identity(x):
        call_count[x] = call_count.get(x, 0) + 1
        return x

    return counting_identity, call_count

min_counting_identity, min_call_count = make_counting_identity()
assert_eq(min("min".elems(), key = min_counting_identity), "i")
assert_eq(min_call_count, {"m": 1, "i": 1, "n": 1})

# max
assert_eq(max("abcdefxyz".elems()), "z")
assert_eq(max("test", "xyz"), "xyz")
assert_eq(max("test", "xyz"), "xyz")
assert_eq(max([1, 2], [5]), [5])
assert_eq(max([-1]), -1)
assert_eq(max([5, 2, 3]), 5)
assert_eq(max({1: 2, -1: 3}), 1)  # a single dict argument is treated as its sequence of keys
assert_eq(max({2: None}), 2)  # a single dict argument is treated as its sequence of keys
assert_eq(max(-1, 2), 2)
assert_eq(max(5, 2, 3), 5)
assert_eq(max(1, 1, 1, 1, 1, 1), 1)
assert_eq(max([1, 1, 1, 1, 1, 1]), 1)
assert_eq(max([None]), None)  # not an error: no comparisons required
assert_fails(lambda: max(1), "type 'int' is not iterable")
assert_fails(lambda: max(), "expected at least one item")
assert_fails(lambda: max([]), "expected at least one item")
assert_fails(lambda: max(1, "2", True), "unsupported comparison: int <=> string")
assert_fails(lambda: max([1, "2", True]), "unsupported comparison: int <=> string")

# max with key
assert_eq(max("aBcDeFXyZ".elems(), key = lambda s: s.lower()), "Z")
assert_eq(max("test", "xyz", key = len), "test")
assert_eq(max([1, 2], [5], key = lambda x: x), [5])
assert_eq(max([-1], key = lambda x: x), -1)
assert_eq(max([5, 2, 3], key = lambda x: x), 5)
assert_eq(max({1: 2, -1: 3}, key = lambda x: x), 1)  # a single dict argument is treated as its sequence of keys
assert_eq(max({2: None}, key = lambda x: x), 2)  # a single dict argument is treated as its sequence of keys
assert_eq(max(-1, 2, key = lambda x: x), 2)
assert_eq(max(5, 2, 3, key = lambda x: x), 5)
assert_eq(max(1, 1, 1, 1, 1, 1, key = lambda x: -x), 1)
assert_eq(max([1, 1, 1, 1, 1, 1], key = lambda x: -x), 1)
assert_fails(lambda: max(1, key = lambda x: x), "type 'int' is not iterable")
assert_fails(lambda: max(key = lambda x: x), "expected at least one item")
assert_fails(lambda: max([], key = lambda x: x), "expected at least one item")
assert_fails(lambda: max([1], ["2"], [True], key = lambda x: x[0]), "unsupported comparison: (int <=> string|string <=> int)")
assert_fails(lambda: max([[1], ["2"], [True]], key = lambda x: x[0]), "unsupported comparison: (int <=> string|string <=> int)")

# verify max with key chooses first value with minimal key
assert_eq(max(1, -1, -2, 2, key = abs), -2)
assert_eq(max([1, -1, -2, 2], key = abs), -2)

# max with failing key
assert_fails(lambda: max(0, 1, 2, 3, 4, key = lambda i: "xyz".elems()[i]), "index out of range \\(index is 3, but sequence has 3 elements\\)")
assert_fails(lambda: max([0, 1, 2, 3, 4], key = lambda i: "xyz".elems()[i]), "index out of range \\(index is 3, but sequence has 3 elements\\)")

# max with non-callable key
assert_fails(lambda: max(1, 2, 3, key = "hello"), "parameter 'key' got value of type 'string', want 'callable or NoneType'")
assert_fails(lambda: max([1, 2, 3], key = "hello"), "parameter 'key' got value of type 'string', want 'callable or NoneType'")

# verify max with key invokes key callback exactly once per item
max_counting_identity, max_call_count = make_counting_identity()
assert_eq(max("max".elems(), key = max_counting_identity), "x")
assert_eq(max_call_count, {"m": 1, "a": 1, "x": 1})
