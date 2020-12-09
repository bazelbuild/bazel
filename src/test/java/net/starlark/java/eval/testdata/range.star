assert_eq(list(range(5)), [0, 1, 2, 3, 4])
assert_eq(list(range(0)), [])
assert_eq(list(range(1)), [0])
assert_eq(list(range(-2)), [])
assert_eq(list(range(-3, 2)), [-3, -2, -1, 0, 1])
assert_eq(list(range(3, 2)), [])
assert_eq(list(range(3, 3)), [])
assert_eq(list(range(3, 4)), [3])
assert_eq(list(range(3, 5)), [3, 4])
assert_eq(list(range(-3, 5, 2)), [-3, -1, 1, 3])
assert_eq(list(range(-3, 6, 2)), [-3, -1, 1, 3, 5])
assert_eq(list(range(5, 0, -1)), [5, 4, 3, 2, 1])
assert_eq(list(range(5, 0, -10)), [5])
assert_eq(list(range(0, -3, -2)), [0, -2])

int32_range_range = range(-1 << 31, (1 << 31) - 1)
assert_eq(len(int32_range_range), 4294967295)
assert_((-1 << 31) - 1 not in int32_range_range)
assert_(-1 << 31 in int32_range_range)
assert_((1 << 31) - 2 in int32_range_range)
assert_((1 << 31) - 1 not in int32_range_range)

---
range(2, 3, 0) ### step cannot be 0

