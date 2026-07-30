# tests of tuples

# index
t = (1, "2", 3)
assert_eq(t[0], 1)
assert_eq(t[1], "2")
assert_eq(t[2], 3)

# index on singleton
s = (1,)
assert_eq(s[0], 1)

# concatenation
assert_eq(() + (1, "a"), (1, "a"))
assert_eq((1, "a") + (), (1, "a"))
assert_eq((1,) + ("a",), (1, "a"))
assert_eq((1,) + ("a", 2), (1, "a", 2))
assert_eq((1, 2) + ("a",), (1, 2, "a"))

# comparison
assert_eq((1, "two", [3, 4]), (1, "two", [3, 4]))
assert_eq(("one",), ("one",))
assert_eq((), ())
assert_(() != (1,))
assert_(() != (1, 2))
assert_((1,) != (1, 2))
assert_((1) != (1,))

# repeat
assert_eq((1, 2, 3) * 3, (1, 2, 3, 1, 2, 3, 1, 2, 3))
assert_eq(3 * (1, 2, 3), (1, 2, 3, 1, 2, 3, 1, 2, 3))
assert_eq(3 * (1,), (1, 1, 1))
assert_eq((1,) * 3, (1, 1, 1))
assert_eq((1, 2, 3) * -1, ())
assert_eq((1, 2, 3) * 0, ())
assert_fails(lambda: (1,) * (1 << 35), "got 34359738368 for repeat, want value in signed 32-bit range")
