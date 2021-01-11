# list comprehensions

# The first for clause is resolved outside the comprehension block.
x = [[1, 2]]  # x₀
assert_eq([x for x in x for y in x], [[1, 2], [1, 2]])
#          x₁    x₁   x₀         x₁

# The scope of z (bound in loop 3) includes the operand of loop 2,
# permitting a forward reference.
assert_eq([1 // 0 for x in [] for y in z for z in ()], [])

# Referring to it on the first iteration (before it is bound) is an error.
assert_fails(
    lambda: [1 // 0 for x in [1] for y in z for z in ()],
    "local variable 'z' is referenced before assignment",
)

# In this example there is a static forward reference
# and a dynamic loop-carried dependence.
assert_eq([y for x in (0, 1) for y in ([y] if x else [3])], [3, 3])
