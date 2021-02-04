# insert

foo = ["a", "b"]

foo.insert(0, "c")
assert_eq(foo, ["c", "a", "b"])

foo.insert(1, "d")
assert_eq(foo, ["c", "d", "a", "b"])

foo.insert(4, "e")
assert_eq(foo, ["c", "d", "a", "b", "e"])

foo.insert(-10, "f")
assert_eq(foo, ["f", "c", "d", "a", "b", "e"])

foo.insert(10, "g")
assert_eq(foo, ["f", "c", "d", "a", "b", "e", "g"])

# check that list() returns new mutable object
a = list()
b = list()
a.insert(0, 1)

assert_eq(a, [1])
assert_eq(b, [])

assert_fails(lambda: (1, 2).insert(3), "'tuple' value has no field or method 'insert'")
---
# append

foo = ["a", "b"]
foo.append("c")
assert_eq(foo, ["a", "b", "c"])
foo.append("d")
assert_eq(foo, ["a", "b", "c", "d"])

bar = [1, 2]
assert_eq(bar.append([3, 4]), None)
assert_eq(bar, [1, 2, [3, 4]])

assert_fails(lambda: (1, 2).append(3), "'tuple' value has no field or method 'append'")
---
# extend

foo = ["a", "b"]
assert_eq(foo.extend(["c", "d"]), None)
assert_eq(foo.extend(("e", "f")), None)
assert_eq(foo.extend({"g": None}), None)
assert_eq(foo, ["a", "b", "c", "d", "e", "f", "g"])

assert_fails(lambda: (1, 2).extend([3, 4]), "'tuple' value has no field or method 'extend'")
assert_fails(lambda: [1, 2].extend(3), "type 'int' is not iterable")
assert_fails(lambda: [].extend(range((1 << 31) - 1)), "excessive capacity requested")

---
# remove

foo = ["a", "b", "c", "b"]

foo.remove("b")
assert_eq(foo, ["a", "c", "b"])

foo.remove("c")
assert_eq(foo, ["a", "b"])

foo.remove("a")
assert_eq(foo, ["b"])

foo.remove("b")
assert_eq(foo, [])

assert_fails(lambda: (1, 2).remove(3), "'tuple' value has no field or method 'remove'")
assert_fails(lambda: [1, 2].remove(3), "item 3 not found in list")

# pop

li1 = [2, 3, 4]
assert_eq(li1.pop(), 4)
assert_eq(li1, [2, 3])

li2 = [2, 3, 4]
assert_eq(li2.pop(-2), 3)
assert_eq(li2, [2, 4])

li3 = [2, 3, 4]
assert_eq(li3.pop(1), 3)
assert_eq(li3, [2, 4])

assert_fails(lambda: [1, 2].pop(3), "index out of range \\(index is 3, but sequence has 2 elements\\)")
assert_fails(lambda: (1, 2).pop(), "'tuple' value has no field or method 'pop'")
---

# clear

foo = [1, 2, 3, 4]
foo.clear()
assert_eq(foo, [])

assert_eq(["a", "b"].clear(), None)
