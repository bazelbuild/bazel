# creation

foo = {'a': 1, 'b': [1, 2]}
bar = dict(a=1, b=[1, 2])
baz = dict({'a': 1, 'b': [1, 2]})

assert_eq(foo, bar)
assert_eq(foo, baz)

# get/setdefault

assert_eq(foo.get('a'), 1)
assert_eq(bar.get('b'), [1, 2])
assert_eq(baz.get('c'), None)
assert_eq(baz.setdefault('c', 15), 15)
assert_eq(baz.setdefault('c'), 15)
assert_eq(baz.setdefault('c', 20), 15)
assert_eq(baz.setdefault('d'), None)

# items

assert_eq(foo.items(), [('a', 1), ('b', [1, 2])])

# keys

assert_eq(bar.keys(), ['a', 'b'])

# values

assert_eq(baz.values(), [1, [1, 2], 15, None])

# pop/popitem

assert_eq(baz.pop('d'), None)
assert_eq(foo.pop('a'), 1)
assert_eq(bar.popitem(), ('a', 1))
assert_eq(foo, bar)
assert_eq(foo.pop('a', 0), 0)
assert_eq(foo.popitem(), ('b', [1, 2]))

d = {1: 2}
freeze(d)
assert_fails(lambda: d.setdefault(1, 2), "trying to mutate a frozen dict value")
assert_fails(lambda: d.pop("nonexistent", "default"), "trying to mutate a frozen dict value")

assert_fails(lambda: dict().popitem(), "empty dictionary")
assert_fails(lambda: dict(a=2).pop('z'), 'KeyError: "z"')
assert_fails(lambda: {}.pop([], 1), "unhashable type: 'list'")
---
# update

foo = dict()
baz = dict(a=1, b=[1, 2])
bar = dict(b=[1, 2])

foo.update(baz)
bar.update(a=1)
baz.update({'c': 3})
foo.update([('c', 3)])
bar['c'] = 3
quz = dict()
quz.update(bar.items())

assert_eq(foo, bar)
assert_eq(foo, baz)
assert_eq(foo, quz)

d = {"b": 0}
d.update({"a": 1}, b = 2)
d.update({"c": 0}, c = 3)
d.update(d, d = 4)
d.update([("e", 5)])
d.update((["f", 0],), f = 6)

expected = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6}
assert_eq(d, expected)

# dict union operator |

empty_dict = dict()
dict_with_a_b = dict(a=1, b=[1, 2])
dict_with_b = dict(b=[1, 2])
dict_with_other_b = dict(b=[3, 4])

assert_eq(empty_dict | dict_with_a_b, dict_with_a_b)
# Verify iteration order.
assert_eq((empty_dict | dict_with_a_b).items(), dict_with_a_b.items())
assert_eq(dict_with_a_b | empty_dict, dict_with_a_b)
assert_eq((dict_with_a_b | empty_dict).items(), dict_with_a_b.items())
assert_eq(dict_with_b | dict_with_a_b, dict_with_a_b)
assert_eq((dict_with_b | dict_with_a_b).items(), dict(b=[1, 2], a=1).items())
assert_eq(dict_with_a_b | dict_with_b, dict_with_a_b)
assert_eq((dict_with_a_b | dict_with_b).items(), dict_with_a_b.items())
assert_eq(dict_with_b | dict_with_other_b, dict_with_other_b)
assert_eq((dict_with_b | dict_with_other_b).items(), dict_with_other_b.items())
assert_eq(dict_with_other_b | dict_with_b, dict_with_b)
assert_eq((dict_with_other_b | dict_with_b).items(), dict_with_b.items())

assert_eq(empty_dict, dict())
assert_eq(dict_with_b, dict(b=[1,2]))

assert_fails(lambda: dict() | [], "unsupported binary operation")

# dict union assignment operator |=

def test_dict_union_assignment():
    empty_dict = dict()
    empty_dict |= {"a": 1}
    empty_dict |= {"b": 2}
    empty_dict |= {"c": "3", 7: 4}
    empty_dict |= {"b": "5", "e": 6}
    expected_1 = {"a": 1, "b": "5", "c": "3", 7: 4, "e": 6}
    assert_eq(empty_dict, expected_1)
    assert_eq(empty_dict.items(), expected_1.items())

    dict_a = {8: 1, "b": 2}
    dict_b = {"b": 1, "c": 6}
    dict_c = {"d": 7}
    dict_tuple = {(5, "a"): ("c", 8)}
    dict_a |= dict_b
    dict_c |= dict_a
    dict_c |= dict_tuple
    expected_2 = {"d": 7, 8: 1, "b": 1, "c": 6, (5, "a"): ("c", 8)}
    assert_eq(dict_c, expected_2)
    assert_eq(dict_c.items(), expected_2.items())
    assert_eq(dict_b, {"b": 1, "c": 6})

test_dict_union_assignment()

def dict_union_assignment_type_mismatch():
    some_dict = dict()
    some_dict |= []

assert_fails(dict_union_assignment_type_mismatch, "unsupported binary operation")

# creation with repeated keys

d1 = dict([('a', 1)], a=2)
d2 = dict({'a': 1}, a=2)
d3 = dict([('a', 1), ('a', 2)])

assert_eq(d1['a'], 2)
assert_eq(d1, d2)
assert_eq(d1, d3)


# in

assert_eq(1 in {1: 0}, True)
assert_eq(() in {}, False)
assert_eq("a" in dict(a=1), True)

# What's going on here? Functions _are_ hashable.
# 'len in {}' and '{}.get(len, False)' should both successfully evaluate to False.
# TODO(adonovan): clarify spec and fix this test (https://github.com/bazelbuild/starlark/issues/65)

# Starlark functions are already hashable:
def f(): pass
f in {} # no error

# unhashable types
assert_fails(lambda: {} in {}, "unhashable type: 'dict'")
assert_fails(lambda: [] in {}, "unhashable type: 'list'")
assert_fails(lambda: len in {}, "unhashable type: 'builtin_function_or_method'")
assert_fails(lambda: {}.get([]), "unhashable type: 'list'")
assert_fails(lambda: dict().get({}), "unhashable type: 'dict'")
assert_fails(lambda: {1: 2}.get(len), "unhashable type: 'builtin_function_or_method'")

# For composite keys, the error message relates to the
# unhashable subelement of the key, not the key itself.
assert_fails(lambda: {(0, "", True, [0]): None}, "unhashable type: 'list'")
