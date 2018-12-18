# All with empty value
assert_eq(all([]), True)

# All with list
assert_eq(all(['t', 'e']), True)
assert_eq(all([False]), False)
assert_eq(all([True, False]), False)
assert_eq(all([False, False]), False)
assert_eq(all([False, True]), False)
assert_eq(all(['', True]), False)
assert_eq(all([0, True]), False)
assert_eq(all([[], True]), False)
assert_eq(all([True, 't', 1]), True)

# All with dict
assert_eq(all({1 : None}), True)
assert_eq(all({None : 1}), False)

# Any with empty value
assert_eq(any([]), False)

# Any with list
assert_eq(any([False]), False)
assert_eq(any([0]), False)
assert_eq(any(['']), False)
assert_eq(any([[]]), False)
assert_eq(any([True, False]), True)
assert_eq(any([False, False]), False)
assert_eq(any([False, '', 0]), False)
assert_eq(any([False, '', 42]), True)

# Any with dict
assert_eq(any({1 : None, '' : None}), True)
assert_eq(any({None : 1, '' : 2}), False)

---
all(None) ### type 'NoneType' is not iterable
---
any(None) ### type 'NoneType' is not iterable
---
any(1) ### type 'int' is not iterable
---
all(1) ### type 'int' is not iterable
