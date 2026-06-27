# str.join

assert_eq("-".join(["a", "b", "c"]), "a-b-c")
assert_eq(", ".join(["a", "b", "c"]), "a, b, c")
assert_eq("".join(["a", "b", "c"]), "abc")
assert_eq("/".join(["a"]), "a")
assert_eq("/".join([]), "")
assert_eq("/".join([""]), "")
assert_eq("/".join(["", ""]), "/")
assert_eq("/".join(["", "", ""]), "//")

assert_eq(" -> ".join(["a", "b", "c"]), "a -> b -> c")
assert_eq("".join([]), "")

assert_eq("|".join(("a", "b", "c")), "a|b|c")
assert_eq("|".join({"a": 1, "b": 2, "c": 3}), "a|b|c")
assert_eq("|".join({}), "")
assert_eq("|".join([x for x in ["a", "b", "c"]]), "a|b|c")
assert_eq("|".join("abc".elems()), "a|b|c")

assert_eq(type("-".join(["a", "b"])), "string")

assert_fails(
    lambda: ",".join(["a", 1, "c"]),
    "expected string for sequence element 1, got '1' of type int",
)
assert_fails(
    lambda: ",".join([None]),
    "expected string for sequence element 0, got 'None' of type NoneType",
)
assert_fails(
    lambda: ",".join(["a", "b", ["c"]]),
    "expected string for sequence element 2, got .* of type list",
)
assert_fails(
    lambda: ",".join([True]),
    "expected string for sequence element 0, got 'True' of type bool",
)

assert_fails(lambda: ",".join(1), "got value of type 'int', want 'iterable'")
assert_fails(lambda: ",".join(None), "got value of type 'NoneType', want 'iterable'")
assert_fails(lambda: ",".join("abc"), "got value of type 'string', want 'iterable'")
assert_fails(lambda: ",".join(), "missing 1 required positional argument: elements")
