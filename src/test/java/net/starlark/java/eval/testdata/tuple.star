# tests of tuples

assert_eq(() + (1, "a"), (1, "a"))
assert_eq((1, "a") + (), (1, "a"))
assert_eq((1,) + ("a",), (1, "a"))
