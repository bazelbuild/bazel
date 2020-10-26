# floating-point numbers

# type
assert_eq(type(0.0), "float")

# truth
assert_eq(bool(0.0), False)
assert_eq(bool(-1.0e-45), True)
assert_eq(bool(float("NaN")), True)

# not iterable
---
len(0.0) ### float is not iterable
---

# int/float equality
assert_eq(0.0, 0)
assert_eq(0, 0.0)
assert_eq(1.0, 1)
assert_eq(1, 1.0)
assert_(1.23e45 != 1229999999999999973814869011019624571608236031)
assert_(1.23e45 == 1229999999999999973814869011019624571608236032)
assert_(1.23e45 != 1229999999999999973814869011019624571608236033)
assert_(1229999999999999973814869011019624571608236031 != 1.23e45)
assert_(1229999999999999973814869011019624571608236032 == 1.23e45)
assert_(1229999999999999973814869011019624571608236033 != 1.23e45)

# loss of precision
p53 = 1<<53
assert_eq(float(p53-1), p53-1)
assert_eq(float(p53+0), p53+0)
assert_eq(float(p53+1), p53+0) #
assert_eq(float(p53+2), p53+2)
assert_eq(float(p53+3), p53+4) #
assert_eq(float(p53+4), p53+4)
assert_eq(float(p53+5), p53+4) #
assert_eq(float(p53+6), p53+6)
assert_eq(float(p53+7), p53+8) #
assert_eq(float(p53+8), p53+8)

assert_(float(p53+1) != p53+1) # comparisons are exact
assert_eq(float(p53+1) - (p53+1), 0) # arithmetic entails rounding

{123.0: "f", 123: "i"} ### dictionary expression has duplicate key: 123
---
# equal int/float values have same hash
d = {123.0: "x"}
d[123] = "y"
assert_eq(len(d), 1)
assert_eq(d[123.0], "y")

# literals (mostly covered by scanner tests)
assert_eq(str(0.), "0.0")
assert_eq(str(.0), "0.0")
assert_(5.0 != 4.999999999999999)
assert_eq(5.0, 4.9999999999999999) # both literals denote 5.0
assert_eq(1.23e45, 1.23 * 1000000000000000000000000000000000000000000000)
assert_eq(str(1.23e-45 - (1.23 / 1000000000000000000000000000000000000000000000)), "-1.5557538194652854e-61")


nan = float("NaN")
inf = float("+Inf")
neginf = float("-Inf")
negzero = (-1e-323 / 10)


# -- arithmetic --
# +float, -float
assert_eq(+(123.0), 123.0)
assert_eq(-(123.0), -123.0)
assert_eq(-(-(123.0)), 123.0)
assert_eq(+(inf), inf)
assert_eq(-(inf), neginf)
assert_eq(-(neginf), inf)
assert_eq(str(-(nan)), "nan")
# +
assert_eq(1.2e3 + 5.6e7, 5.60012e+07)
assert_eq(1.2e3 + 1, 1201)
assert_eq(1 + 1.2e3, 1201)
assert_eq(str(1.2e3 + nan), "nan")
assert_eq(inf + 0, inf)
assert_eq(inf + 1, inf)
assert_eq(inf + inf, inf)
assert_eq(str(inf + neginf), "nan")
# -
assert_eq(1.2e3 - 5.6e7, -5.59988e+07)
assert_eq(1.2e3 - 1, 1199)
assert_eq(1 - 1.2e3, -1199)
assert_eq(str(1.2e3 - nan), "nan")
assert_eq(inf - 0, inf)
assert_eq(inf - 1, inf)
assert_eq(str(inf - inf), "nan")
assert_eq(inf - neginf, inf)
# *
assert_eq(1.5e6 * 2.2e3, 3.3e9)
assert_eq(1.5e6 * 123, 1.845e+08)
assert_eq(123 * 1.5e6, 1.845e+08)
assert_eq(str(1.2e3 * nan), "nan")
assert_eq(str(inf * 0), "nan")
assert_eq(inf * 1, inf)
assert_eq(inf * inf, inf)
assert_eq(inf * neginf, neginf)
# %
assert_eq(100.0 % 7.0, 2)
assert_eq(100.0 % -7.0, -5) # NB: different from Go / Java
assert_eq(-100.0 % 7.0, 5) # NB: different from Go / Java
assert_eq(-100.0 % -7.0, -2)
assert_eq(-100.0 % 7, 5)
assert_eq(100 % 7.0, 2)
assert_eq(str(1.2e3 % nan), "nan")
assert_eq(str(inf % 1), "nan")
assert_eq(str(inf % inf), "nan")
assert_eq(str(inf % neginf), "nan")
# /
assert_eq(str(100.0 / 7.0), "14.285714285714286")
assert_eq(str(100 / 7.0), "14.285714285714286")
assert_eq(str(100.0 / 7), "14.285714285714286")
assert_eq(str(100.0 / nan), "nan")
# //
assert_eq(100.0 // 7.0, 14)
assert_eq(100 // 7.0, 14)
assert_eq(100.0 // 7, 14)
assert_eq(100.0 // -7.0, -15)
assert_eq(100 // -7.0, -15)
assert_eq(100.0 // -7, -15)
assert_eq(str(1 // neginf), "-0.0")
assert_eq(str(100.0 // nan), "nan")

# -- comparisons --
# NaN
assert_(nan == nan) # \
assert_(nan >= nan) #  unlike Python
assert_(nan <= nan) # /
assert_(not (nan > nan))
assert_(not (nan < nan))
assert_(not (nan != nan)) # unlike Python
# Sort is stable: 0.0 and -0.0 are equal, but they are not permuted.
# Similarly 1 and 1.0.
assert_eq(
    str(sorted([inf, neginf, nan, 1e300, -1e300, 1.0, -1.0, 1, -1, 1e-300, -1e-300, 0, 0.0, negzero, 1e-300, -1e-300])),
    "[-inf, -1e+300, -1.0, -1, -1e-300, -1e-300, 0, 0.0, -0.0, 1e-300, 1e-300, 1.0, 1, 1e+300, +inf, nan]")

# Sort is stable, and its result contains no adjacent x, y such that y > x.
# Note: Python's reverse sort is unstable; see https://bugs.python.org/issue36095.
assert_eq(str(sorted([7, 3, nan, 1, 9])), "[1, 3, 7, 9, nan]")
assert_eq(str(sorted([7, 3, nan, 1, 9], reverse=True)), "[nan, 9, 7, 3, 1]")

# All NaN values compare equal. (Identical objects compare equal.)
nandict = {nan: 1}
nandict[nan] = 2
assert_eq(len(nandict), 1) # (same as Python)
assert_eq(nandict[nan], 2) # (same as Python)

nandict[float('nan')] = 3 # a distinct NaN object
assert_eq(str(nandict), "{nan: 3}") # (Python: {nan: 2, nan: 3})

assert_eq(str({inf: 1, neginf: 2}), "{+inf: 1, -inf: 2}")

# zero
assert_eq(0.0, negzero)

# min/max ordering with NaN (the greatest float value)
assert_eq(max([1, nan, 3]), nan)
assert_eq(max([nan, 2, 3]), nan)
assert_eq(min([1, nan, 3]), 1)
assert_eq(min([nan, 2, 3]), 2)

# -- formatting --
# %d
assert_eq("%d" % 0, "0")
assert_eq("%d" % 0.0, "0")
assert_eq("%d" % 123, "123")
assert_eq("%d" % 123.0, "123")
assert_eq("%d" % 1.23e45, "1229999999999999973814869011019624571608236032")
# (see below for '%d' % NaN/Inf)
assert_eq("%d" % negzero, "0")
# %e
assert_eq("%e" % 0, "0.000000e+00")
assert_eq("%e" % 0.0, "0.000000e+00")
assert_eq("%e" % 123, "1.230000e+02")
assert_eq("%e" % 123.0, "1.230000e+02")
assert_eq("%e" % 1.23e45, "1.230000e+45")
assert_eq("%e" % -1.23e-45, "-1.230000e-45")
assert_eq("%e" % nan, "nan")
assert_eq("%e" % inf, "+inf")
assert_eq("%e" % neginf, "-inf")
assert_eq("%e" % negzero, "-0.000000e+00")
# %f
assert_eq("%f" % 0, "0.000000")
assert_eq("%f" % 0.0, "0.000000")
assert_eq("%f" % 123, "123.000000")
assert_eq("%f" % 123.0, "123.000000")
assert_eq("%f" % 1.23e45, "1230000000000000000000000000000000000000000000.000000")
assert_eq("%f" % -1.23e-45, "-0.000000")
assert_eq("%f" % nan, "nan")
assert_eq("%f" % inf, "+inf")
assert_eq("%f" % neginf, "-inf")
assert_eq("%f" % negzero, "-0.000000")
# %g
assert_eq("%g" % 0, "0.0")
assert_eq("%g" % 0.0, "0.0")
assert_eq("%g" % 123, "123.0")
assert_eq("%g" % 123.0, "123.0")
assert_eq("%g" % 1.110, "1.11")
# The threshold for scientific notation should be 1e+7, not 1e+17.
# TODO(adonovan): implement minimal rounding.
assert_eq("%g" % 1e16, "10000000000000000.0")
assert_eq("%g" % 1e17, "1e+17")
assert_eq("%g" % 1e20, "1e+20")
assert_eq("%g" % 1.23e45, "1.23e+45")
assert_eq("%g" % -1.23e-45, "-1.23e-45")
assert_eq("%g" % nan, "nan")
assert_eq("%g" % inf, "+inf")
assert_eq("%g" % neginf, "-inf")
assert_eq("%g" % negzero, "-0.0")
# str uses %g
assert_eq(str(0.0), "0.0")
assert_eq(str(123.0), "123.0")
assert_eq(str(1.23e45), "1.23e+45")
assert_eq(str(-1.23e-45), "-1.23e-45")
assert_eq(str(nan), "nan")
assert_eq(str(inf), "+inf")
assert_eq(str(neginf), "-inf")
assert_eq(str(negzero), "-0.0")

# -- float function --
# float()
assert_eq(float(), 0.0)
# float(bool)
assert_eq(float(False), 0.0)
assert_eq(float(True), 1.0)
assert_(False != 0.0) # differs from Python
assert_(True != 1.0)
# float(int)
assert_eq(float(0), 0.0)
assert_eq(float(1), 1.0)
assert_eq(float(123), 123.0)
assert_eq(float(123 * 1000000 * 1000000 * 1000000 * 1000000 * 1000000), 1.23e+32)
# float(string)
assert_eq(str(float("NaN")), "nan")
assert_eq(str(float("+NAN")), "nan")
assert_eq(str(float("-nan")), "nan")
assert_eq(str(float("Inf")), "+inf")
assert_eq(str(float("+INF")), "+inf")
assert_eq(str(float("-inf")), "-inf")
assert_eq(str(float("+InFiniTy")), "+inf")
assert_eq(str(float("-iNFiniTy")), "-inf")

# -- int function --
assert_eq(int(0.0), 0)
assert_eq(int(1.0), 1)
assert_eq(int(1.1), 1)
assert_eq(int(0.9), 0)
assert_eq(int(-1.1), -1.0)
assert_eq(int(-1.0), -1.0)
assert_eq(int(-0.9), 0.0)
assert_eq(int(1.23e+32), 123000000000000004979083645550592)
assert_eq(int(-1.23e-32), 0)
assert_eq(int(1.23e-32), 0)

# TODO(adonovan): put these errors in the correct order when we have assert_fails(lambda).
---
"%d" % float("NaN") ### got nan, want a finite number
---
"%d" % float("+Inf") ###  got \+inf, want a finite number
---
"%d" % float("-Inf") ### got -inf, want a finite number
---
1 // 0.0 ### integer division by zero
---
1.0 // 0.0 ### integer division by zero
---
1 // 0 ### integer division by zero
---
1 / 0.0 ### floating-point division by zero
---
1.0 % 0.0 ### floating-point modulo by zero
---
1.0 % 0 ### floating-point modulo by zero
---
1 % 0.0 ### floating-point modulo by zero
---
# A float value may not be used where an integer is required,
# even if its value is an integer.
range(3)[1.0] ### got float for sequence index, want int
---
float("one point two") ### invalid float literal: one point two
---
float("1.2.3") ### invalid float literal: 1.2.3
---
int(float("+Inf")) ### can't convert float \+inf to int
---
int(float("-Inf")) ### can't convert float -inf to int
---
int(float("NaN")) ### can't convert float nan to int
---
float(123 << 500 << 500 << 50) ### int too large to convert to float
---
float(-123 << 500 << 500 << 50) ### int too large to convert to float
---
float(str(-123 << 500 << 500 << 50)) ### floating-point number too large
---
{float('nan'): 1, float('nan'): 2} ### duplicate key
---
# implicit float(int) conversion in float x int fails if too large
(1<<500<<500<<500) + 0.0 ### int too large to convert to float
---
0.0 + (1<<500<<500<<500) ### int too large to convert to float
---
(1<<500<<500<<500) - 0.0 ### int too large to convert to float
---
0.0 - (1<<500<<500<<500) ### int too large to convert to float
---
(1<<500<<500<<500) * 1.0 ### int too large to convert to float
---
1.0 * (1<<500<<500<<500) ### int too large to convert to float
---
(1<<500<<500<<500) / 1.0 ### int too large to convert to float
---
1.0 / (1<<500<<500<<500) ### int too large to convert to float
---
(1<<500<<500<<500) // 1.0 ### int too large to convert to float
---
1.0 // (1<<500<<500<<500) ### int too large to convert to float
---
(1<<500<<500<<500) % 1.0 ### int too large to convert to float
---
1.0 % (1<<500<<500<<500) ### int too large to convert to float
