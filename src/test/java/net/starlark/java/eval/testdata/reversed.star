# lists

assert_eq(reversed([]), [])
assert_eq(reversed("a".elems()), ["a"])
assert_eq(reversed("abc".elems()), ["c", "b", "a"])
assert_eq(reversed("__test  ".elems()), [" ", " ", "t", "s", "e", "t", "_", "_"])
assert_eq(reversed("bbb".elems()), ["b", "b", "b"])
assert_eq(reversed({"one": 1, "two": 2, "three": 3}), ["three", "two", "one"])
assert_fails(lambda: reversed(None), "got .* 'NoneType', want 'iterable'")
assert_fails(lambda: reversed(1), "got .* 'int', want 'iterable'")

x = ["a", "b"]
y = reversed(x)
y.append("c")
assert_eq(y, ["b", "a", "c"])
assert_eq(x, ["a", "b"])

def reverse_equivalence(inp):
    assert_eq(reversed(inp), inp[::-1])
    assert_eq(reversed(reversed(inp)), inp)

reverse_equivalence([])
reverse_equivalence([1])
reverse_equivalence(["a", "b"])
