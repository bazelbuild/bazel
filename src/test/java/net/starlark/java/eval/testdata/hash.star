# hash

assert_eq(hash(""), 0)
assert_eq(hash("a"), 97)
assert_eq(hash("ab"), 3105)
assert_eq(hash("abc"), 96354)
assert_eq(hash("A"), 65)

assert_eq(hash("hello world"), hash("hello world"))

s1 = "x" + "yz"
s2 = "xyz"
assert_eq(hash(s1), hash(s2))

assert_(hash("foo") != hash("bar"))
assert_(hash("a") != hash("b"))

assert_eq(type(hash("anything")), "int")

assert_fails(lambda: hash(1), "got value of type 'int', want 'string'")
assert_fails(lambda: hash(1.5), "got value of type 'float', want 'string'")
assert_fails(lambda: hash(True), "got value of type 'bool', want 'string'")
assert_fails(lambda: hash(None), "got value of type 'NoneType', want 'string'")
assert_fails(lambda: hash(["a"]), "got value of type 'list', want 'string'")
assert_fails(lambda: hash(("a",)), "got value of type 'tuple', want 'string'")
assert_fails(lambda: hash({"a": 1}), "got value of type 'dict', want 'string'")
assert_fails(lambda: hash(), "missing 1 required positional argument: value")
assert_fails(lambda: hash("a", "b"), "accepts no more than 1 positional argument")
