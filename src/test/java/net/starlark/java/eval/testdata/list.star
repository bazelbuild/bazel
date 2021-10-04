# list tests
# TODO(adonovan): duplicate/share these tests for tuple where applicable.

# index
l = [1, "2", 3]
assert_eq(l[0], 1)
assert_eq(l[1], "2")
assert_eq(l[2], 3)

# index out of bounds
assert_fails(
    lambda: ["a", "b", "c"][3],
    "index out of range \\(index is 3, but sequence has 3 elements\\)",
)
assert_fails(
    lambda: ["a", "b", "c"][10],
    "index out of range \\(index is 10, but sequence has 3 elements\\)",
)
assert_fails(
    lambda: [][0],
    "index out of range \\(index is 0, but sequence has 0 elements\\)",
)

# negative indices
m = ["a", "b", "c"]
assert_eq(m[0], "a")
assert_eq(m[-1], "c")
assert_eq(m[-2], "b")
assert_eq(m[-3], "a")
assert_fails(lambda: m[-4], "index out of range \\(index is -4, but sequence has 3 elements\\)")
assert_fails(lambda: [][-1], "index out of range \\(index is -1, but sequence has 0 elements\\)")

# len
assert_eq(len([42, "hello, world", []]), 3)

# truth
assert_eq(8 if [1, 2, 3] else 9, 8)
assert_eq(8 if [] else 9, 9)

# concat
assert_eq([1, 2] + [3, 4], [1, 2, 3, 4])
assert_eq(len([1, 2] + [3, 4]), 4)
assert_eq(str([1, 2] + [3, 4]), "[1, 2, 3, 4]")
assert_eq(1 if ([1, 2] + [3, 4]) else 0, 1)
assert_eq(1 if ([] + []) else 0, 0)

h = [1] + (([2] + [3, 4]) + [5])
assert_eq(h[0], 1)
assert_eq(h[1], 2)
assert_eq(h[2], 3)
assert_eq(h[3], 4)
assert_eq(h[4], 5)

# comparison
assert_eq([1, 2, 3, 4], [1, 2] + [3, 4])
assert_([1, 2, 3, 4] != (1, 2, 3, 4))
assert_([1, 2] != [1, 2, 3])
assert_eq([], [])

# repeat
assert_eq([1, 2, 3] * 3, [1, 2, 3, 1, 2, 3, 1, 2, 3])
assert_eq(3 * [1, 2, 3], [1, 2, 3, 1, 2, 3, 1, 2, 3])
assert_eq([1, 2, 3] * -1, [])
assert_eq([1, 2, 3] * 0, [])
assert_fails(lambda: [1] * (1 << 35), "got 34359738368 for repeat, want value in signed 32-bit range")
assert_fails(lambda: [1, 2, 3] * (1 << 30), "excessive repeat \\(3 \\* 1073741824 elements\\)")
