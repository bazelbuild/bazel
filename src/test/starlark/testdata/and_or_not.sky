assert_eq(8 or 9, 8)
assert_eq(0 or 9, 9)
assert_eq(8 and 9, 9)
assert_eq(0 and 9, 0)

assert_eq(1 and 2 or 3, 2)
assert_eq(0 and 2 or 3, 3)
assert_eq(1 and 0 or 3, 3)

assert_eq(1 or 2 and 3, 1)
assert_eq(0 or 2 and 3, 3)
assert_eq(0 or 0 and 3, 0)
assert_eq(1 or 0 and 3, 1)

assert_eq(None and 1, None)
assert_eq("" or 9, 9)
assert_eq("abc" or 9, "abc")

# check that fail() is not evaluated
assert_eq(8 or fail("do not execute"), 8)
assert_eq(0 and fail("do not execute"), 0)

assert_eq(not 1, False)
assert_eq(not "", True)
assert_eq(not not 1, True)

assert_eq(not 0 + 0, True)
assert_eq(not 2 - 1, False)

assert_eq(not (0 and 0), True)
assert_eq(not (1 or 0), False)

assert_eq(0 and not 0, 0)
assert_eq(not 0 and 0, 0)

assert_eq(1 and not 0, True)
assert_eq(not 0 or 0, True)

assert_eq(not 1 or 0, 0)
assert_eq(not 1 or 1, 1)

assert_eq(not [], True)
assert_eq(not {"a": 1}, False)
