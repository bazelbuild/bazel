# tests of cyclic data structures

# list containing itself
x = [1, 2, 3]
x[1] = x
assert_eq(repr(x), "[1, ..., 3]")
---
# dict containing itself
x = {1: 1, 2: 2}
x[1] = x
assert_eq(repr(x), "{1: ..., 2: 2}")
---
# 3-node cycle
x = [1, 2, 3]
y = [4, x, 6]
z = {1: x, 2: y}
x[1] = z
assert_eq(repr(x), "[1, {1: ..., 2: [4, ..., 6]}, 3]")
