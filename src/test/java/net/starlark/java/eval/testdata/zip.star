# zip

assert_eq(zip(), [])

assert_eq(zip([1, 2, 3]), [(1,), (2,), (3,)])
assert_eq(zip(()), [])
assert_eq(zip([]), [])
assert_eq(zip(range(3)), [(0,), (1,), (2,)])

assert_eq(zip([1, 2, 3], [4, 5, 6]), [(1, 4), (2, 5), (3, 6)])
assert_eq(zip((1, 2), (3, 4), (5, 6)), [(1, 3, 5), (2, 4, 6)])

assert_eq(zip([1, 2, 3], [4, 5]), [(1, 4), (2, 5)])
assert_eq(zip([1], [2, 3], [4, 5, 6]), [(1, 2, 4)])
assert_eq(zip([], [1, 2]), [])
assert_eq(zip([1, 2], []), [])

assert_eq(zip([1, 2], (3, 4)), [(1, 3), (2, 4)])
assert_eq(zip({"a": 1, "b": 2}, [10, 20]), [("a", 10), ("b", 20)])
assert_eq(zip("ab".elems(), "cd".elems()), [("a", "c"), ("b", "d")])
assert_eq(zip(range(2), [3, 4, 5]), [(0, 3), (1, 4)])

z = zip([1], [2])
assert_eq(type(z), "list")
assert_eq(type(z[0]), "tuple")

src1 = [1, 2, 3]
src2 = ["a", "b", "c"]
out = zip(src1, src2)
out.append(("d", 4))
assert_eq(out, [(1, "a"), (2, "b"), (3, "c"), ("d", 4)])
assert_eq(src1, [1, 2, 3])
assert_eq(src2, ["a", "b", "c"])

assert_fails(lambda: zip("ab"), "type 'string' is not iterable")
assert_fails(lambda: zip([1, 2], "ab"), "type 'string' is not iterable")
assert_fails(lambda: zip(1), "type 'int' is not iterable")
assert_fails(lambda: zip([1, 2], 3), "type 'int' is not iterable")
assert_fails(lambda: zip(None), "type 'NoneType' is not iterable")
assert_fails(lambda: zip(args = [1, 2]), "got unexpected keyword argument 'args'")
