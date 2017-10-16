# == operator
assert_eq(1 == 1, True)
assert_eq(1 == 2, False)
assert_eq('hello' == 'hel' + 'lo', True)
assert_eq('hello' == 'bye', False)
assert_eq(None == None, True)
assert_eq([1, 2] == [1, 2], True)
assert_eq([1, 2] == [2, 1], False)
assert_eq({'a': 1, 'b': 2} == {'b': 2, 'a': 1}, True)
assert_eq({'a': 1, 'b': 2} == {'a': 1}, False)
assert_eq({'a': 1, 'b': 2} == {'a': 1, 'b': 2, 'c': 3}, False)
assert_eq({'a': 1, 'b': 2} == {'a': 1, 'b': 3}, False)

# != operator
assert_eq(1 != 1, False)
assert_eq(1 != 2, True)
assert_eq('hello' != 'hel' + 'lo', False)
assert_eq('hello' != 'bye', True)
assert_eq([1, 2] != [1, 2], False)
assert_eq([1, 2] != [2, 1], True)
assert_eq({'a': 1, 'b': 2} != {'b': 2, 'a': 1}, False)
assert_eq({'a': 1, 'b': 2} != {'a': 1}, True)
assert_eq({'a': 1, 'b': 2} != {'a': 1, 'b': 2, 'c': 3}, True)
assert_eq({'a': 1, 'b': 2} != {'a': 1, 'b': 3}, True);

# equality precedence
assert_eq(1 + 3 == 2 + 2, True)
assert_eq(not 1 == 2, True)
assert_eq(not 1 != 2, False)
assert_eq(2 and 3 == 3 or 1, True)
assert_eq(2 or 3 == 3 and 1, 2);

# < operator
assert_eq(1 <= 1, True)
assert_eq(1 < 1, False)
assert_eq('a' <= 'b', True)
assert_eq('c' < 'a', False);

# <= and < operators
assert_eq(1 <= 1, True)
assert_eq(1 < 1, False)
assert_eq('a' <= 'b', True)
assert_eq('c' < 'a', False);

# >= and > operators
assert_eq(1 >= 1, True)
assert_eq(1 > 1, False)
assert_eq('a' >= 'b', False)
assert_eq('c' > 'a', True);

# list/tuple comparison
assert_eq([] < [1], True)
assert_eq([1] < [1, 1], True)
assert_eq([1, 1] < [1, 2], True)
assert_eq([1, 2] < [1, 2, 3], True)
assert_eq([1, 2, 3] <= [1, 2, 3], True)

assert_eq(['a', 'b'] > ['a'], True)
assert_eq(['a', 'b'] >= ['a'], True)
assert_eq(['a', 'b'] < ['a'], False)
assert_eq(['a', 'b'] <= ['a'], False)

assert_eq(('a', 'b') > ('a', 'b'), False)
assert_eq(('a', 'b') >= ('a', 'b'), True)
assert_eq(('a', 'b') < ('a', 'b'), False)
assert_eq(('a', 'b') <= ('a', 'b'), True)

assert_eq([[1, 1]] > [[1, 1], []], False)
assert_eq([[1, 1]] < [[1, 1], []], True)
