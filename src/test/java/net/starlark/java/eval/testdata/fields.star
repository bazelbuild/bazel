# struct-like field access

# TODO(adonovan): support setattr.

## immutable

x = struct(a = 1, b = 2)
assert_eq(str(x), "struct(a = 1, b = 2)")
assert_eq(type(x), "struct")
assert_eq(x.a, 1)
assert_eq(x.b, 2)

assert_eq(dir(x), ["a", "b"])
assert_eq(getattr(x, "a"), 1)
assert_eq(hasattr(x, "a"), True)
assert_eq(hasattr(x, "c"), False)
assert_eq(getattr(x, "c", 3), 3)  # missing => default

x.c ### 'struct' value has no field or method 'c'
---
x = struct(a = 1, b = 2)
getattr(x, "c") ### 'struct' value has no field or method 'c'
---
x = struct(a = 1, b = 2)
x.c = 3 ### struct value does not support field assignment
---

## mutable

y = mutablestruct(a = 1, b = 2)
assert_eq(str(y), "mutablestruct(a = 1, b = 2)")
assert_eq(type(y), "mutablestruct")
assert_eq(y.a, 1)
assert_eq(getattr(y, "a"), 1)
assert_eq(y.b, 2)
assert_eq(dir(y), ["a", "b"])
y.b = -2  # update existing field
y.c = 3  # set new field

assert_eq(str(y), "mutablestruct(a = 1, b = -2, c = 3)")
y.c = "bad"  ### bad field value
