# repr

assert_eq(repr(None), "None")
assert_eq(repr(True), "True")
assert_eq(repr(False), "False")

assert_eq(repr(0), "0")
assert_eq(repr(42), "42")
assert_eq(repr(-7), "-7")
assert_eq(repr(1 << 64), "18446744073709551616")
assert_eq(repr(-(1 << 64)), "-18446744073709551616")

assert_eq(repr(0.0), "0.0")
assert_eq(repr(-0.0), "-0.0")
assert_eq(repr(1.0), "1.0")
assert_eq(repr(-1.5), "-1.5")
assert_eq(repr(float("+inf")), "+inf")
assert_eq(repr(float("-inf")), "-inf")
assert_eq(repr(float("nan")), "nan")

assert_eq(repr(""), '""')
assert_eq(repr("abc"), '"abc"')
assert_eq(repr('it\'s "ok"'), '"it\'s \\"ok\\""')
assert_eq(repr("a\\b"), '"a\\\\b"')
assert_eq(repr("a\nb"), '"a\\nb"')
assert_eq(repr("a\tb"), '"a\\tb"')
assert_eq(repr("a\rb"), '"a\\rb"')
# Source uses octal escapes (Starlark has no \x escape); repr output uses \xNN.
assert_eq(repr("\0"), '"\\x00"')
assert_eq(repr("\1\037"), '"\\x01\\x1f"')

assert_eq(type(repr(123)), "string")
assert_eq(type(repr(None)), "string")

assert_eq(repr(()), "()")
assert_eq(repr((1,)), "(1,)")
assert_eq(repr((1, 2)), "(1, 2)")
assert_eq(repr((1, "x", None)), '(1, "x", None)')

assert_eq(repr([]), "[]")
assert_eq(repr([1]), "[1]")
assert_eq(repr([1, 2, 3]), "[1, 2, 3]")
assert_eq(repr(["a", "b"]), '["a", "b"]')

assert_eq(repr({}), "{}")
assert_eq(repr({1: 2}), "{1: 2}")
assert_eq(repr({"a": 1, "b": 2}), '{"a": 1, "b": 2}')

assert_eq(repr(set()), "set()")
assert_eq(repr(set([1])), "set([1])")
assert_eq(repr(set([1, 2, 3])), "set([1, 2, 3])")

assert_eq(repr(range(3)), "range(0, 3)")
assert_eq(repr(range(1, 5)), "range(1, 5)")
assert_eq(repr(range(0, 10, 2)), "range(0, 10, 2)")

assert_eq(repr([(1, 2), (3, 4)]), "[(1, 2), (3, 4)]")
assert_eq(repr({"a": [1, 2]}), '{"a": [1, 2]}')
assert_eq(repr([{"a": 1}, {"b": 2}]), '[{"a": 1}, {"b": 2}]')
assert_eq(repr([None, True, False, 0, ""]), '[None, True, False, 0, ""]')

def myfn():
    pass

assert_eq(repr(myfn), "<function myfn>")
assert_eq(repr(lambda: 0), "<function lambda>")

assert_fails(lambda: repr(), "missing 1 required positional argument: x")
assert_fails(lambda: repr(1, 2), "accepts no more than 1 positional argument")
