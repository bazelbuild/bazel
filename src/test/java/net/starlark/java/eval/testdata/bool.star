# bool

assert_eq(bool(), False)

assert_eq(bool(True), True)
assert_eq(bool(False), False)
assert_eq(type(bool(True)), "bool")
assert_eq(type(bool(0)), "bool")
assert_eq(type(bool()), "bool")

assert_eq(bool(None), False)

assert_eq(bool(0), False)
assert_eq(bool(1), True)
assert_eq(bool(-1), True)
assert_eq(bool(1 << 100), True)
assert_eq(bool(-(1 << 100)), True)

assert_eq(bool(0.0), False)
assert_eq(bool(-0.0), False)
assert_eq(bool(0.5), True)
assert_eq(bool(-0.5), True)
assert_eq(bool(float("inf")), True)
assert_eq(bool(float("-inf")), True)
assert_eq(bool(float("nan")), True)

assert_eq(bool(""), False)
assert_eq(bool(" "), True)
assert_eq(bool("0"), True)
assert_eq(bool("False"), True)

assert_eq(bool([]), False)
assert_eq(bool([0]), True)
assert_eq(bool([False]), True)
assert_eq(bool(()), False)
assert_eq(bool((0,)), True)
assert_eq(bool({}), False)
assert_eq(bool({0: 0}), True)
assert_eq(bool(set()), False)
assert_eq(bool(set([0])), True)

assert_eq(bool(range(0)), False)
assert_eq(bool(range(0, 0)), False)
assert_eq(bool(range(1)), True)
assert_eq(bool(range(-5, 5)), True)

assert_eq(bool(struct()), True)
assert_eq(bool(struct(a = 1)), True)

assert_eq(sorted([1, 0, 2, 0, 3], key = bool), [0, 0, 1, 2, 3])

assert_fails(lambda: bool(1, 2), "accepts no more than 1 positional argument")
