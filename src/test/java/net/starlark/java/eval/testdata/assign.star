# tests of assignment

# computation in a[...]=x expression.
a = [0, 1, 2, 3, 4, 5]
a[[i for i in range(6) if i == 2][0]] = "z"
assert_eq(a, [0, 1, "z", 3, 4, 5])
