# abs

assert_eq(abs(0), 0)
assert_eq(abs(1), 1)
assert_eq(abs(-1), 1)
assert_eq(abs(123), 123)
assert_eq(abs(-123), 123)
assert_eq(type(abs(-7)), "int")
assert_eq(type(abs(7)), "int")

# big ints
big = 1 << 100
assert_eq(abs(big), big)
assert_eq(abs(-big), big)
assert_eq(abs(-(1 << 63)), 1 << 63)
assert_eq(abs(-(1 << 31)), 1 << 31)
assert_eq(abs(-2147483648), 2147483648)
assert_eq(abs(-9223372036854775808), 9223372036854775808)

# float
assert_eq(abs(0.0), 0.0)
assert_eq(abs(-0.0), 0.0)
assert_eq(abs(1.5), 1.5)
assert_eq(abs(-1.5), 1.5)
assert_eq(abs(-3.14), 3.14)
assert_eq(type(abs(1.0)), "float")
assert_eq(type(abs(-1.0)), "float")

inf = float("+inf")
neginf = float("-inf")
nan = float("nan")
assert_eq(abs(inf), inf)
assert_eq(abs(neginf), inf)
assert_eq(str(abs(nan)), "nan")

assert_fails(lambda: abs(True), "got value of type 'bool', want 'int or float'")
assert_fails(lambda: abs(False), "got value of type 'bool', want 'int or float'")
assert_fails(lambda: abs("1"), "got value of type 'string', want 'int or float'")
assert_fails(lambda: abs(None), "got value of type 'NoneType', want 'int or float'")
assert_fails(lambda: abs([1]), "got value of type 'list', want 'int or float'")
assert_fails(lambda: abs(), "missing 1 required positional argument: x")
assert_fails(lambda: abs(1, 2), "accepts no more than 1 positional argument")
