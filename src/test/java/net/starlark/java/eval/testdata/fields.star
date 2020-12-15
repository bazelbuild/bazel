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

assert_fails(lambda: x.c, "'struct' value has no field or method 'c'")
assert_fails(lambda: getattr(x, "c"), "'struct' value has no field or method 'c'")

def set_c(x, val):
    x.c = val

assert_fails(lambda: set_c(x, 3), "struct value does not support field assignment")
assert_fails(lambda: getattr(x, "c"), "'struct' value has no field or method 'c'")

def incr_c(x, val):
    x.c += val

assert_fails(lambda: incr_c(x, 3), "'struct' value has no field or method 'c'")

def incr_b(x, val):
    x.b += val

assert_fails(lambda: incr_b(x, 3), "struct value does not support field assignment")

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
y.c += 4  # augmented field assignment

assert_eq(str(y), "mutablestruct(a = 1, b = -2, c = 7)")
assert_fails(lambda: set_c(y, "bad"), "bad field value")

---
# Test of correct error location.
x = struct(a = "")
x.a \
  -= 1 ### unsupported binary operation
