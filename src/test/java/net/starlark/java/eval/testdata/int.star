# Tests of Starlark 'int'

# basic arithmetic
assert_eq(0 - 1, -1)
assert_eq(1 + 1, 2)
assert_eq(5 + 7, 12)
assert_eq(5 * 7, 35)
assert_eq(5 - 7, -2)

# big numbers
assert_eq(+1 << 32, +0x100000000)
assert_eq(-1 << 32, -0x100000000)
assert_eq(+1 << 64, +0x10000000000000000)
assert_eq(-1 << 64, -0x10000000000000000)
assert_eq(+1 << 128, +0x100000000000000000000000000000000)
assert_eq(-1 << 128, -0x100000000000000000000000000000000)
assert_eq((1 << 128) // (1 << 127), 2)

# Not using << to define constants because we are testing <<
maxint = 0x7fffffff  # (1<<31) - 1
maxlong = 0x7fffffffffffffff  # (1<<63) - 1
minint = -0x80000000  # -1 << 31
minlong = -0x8000000000000000  # -1 << 63

# size boundaries
assert_eq(maxint + 1, 0x80000000)
assert_eq(maxlong + 1, 0x8000000000000000)
assert_eq(minint - 1, -0x80000001)
assert_eq(minlong - 1, -0x8000000000000001)

# str(int)
assert_eq(str(0), "0")
assert_eq(str(1), "1")
assert_eq(str(1 << 24), "16777216")
assert_eq(str(1 << 48), "281474976710656")
assert_eq(str(1 << 96), "79228162514264337593543950336")
assert_eq(str(-1 << 24), "-16777216")
assert_eq(str(-1 << 48), "-281474976710656")
assert_eq(str(-1 << 96), "-79228162514264337593543950336")

# %d formatting
assert_eq("%d" % 255, "255")
assert_eq("%d" % (1 << 32), "4294967296")
assert_eq("%d" % (1 << 64), "18446744073709551616")
assert_eq("%d" % (1 << 128), "340282366920938463463374607431768211456")
assert_eq("%d" % (-1 << 128), "-340282366920938463463374607431768211456")

# %x formatting
assert_eq("%x" % 255, "ff")
assert_eq("%x" % (1 << 32), "100000000")
assert_eq("%x" % (1 << 64), "10000000000000000")
assert_eq("%x" % (1 << 128), "100000000000000000000000000000000")
assert_eq("%x" % (-1 << 128), "-100000000000000000000000000000000")
assert_fails(lambda: "%x" % "1", "got string for '%x' format, want int or float")

# %X formatting
assert_eq("%X" % 255, "FF")
assert_eq("%X" % (1 << 32), "100000000")
assert_eq("%X" % (1 << 64), "10000000000000000")
assert_eq("%X" % (1 << 128), "100000000000000000000000000000000")
assert_eq("%X" % (-1 << 128), "-100000000000000000000000000000000")
assert_fails(lambda: "%X" % "1", "got string for '%X' format, want int or float")

# %o formatting
assert_eq("%o" % 255, "377")
assert_eq("%o" % (1 << 32), "40000000000")
assert_eq("%o" % (1 << 64), "2000000000000000000000")
assert_eq("%o" % (1 << 128), "4000000000000000000000000000000000000000000")
assert_eq("%o" % (-1 << 128), "-4000000000000000000000000000000000000000000")
assert_fails(lambda: "%o" % "1", "got string for '%o' format, want int or float")

# truth
assert_(not 0)
assert_(1)
assert_(-1)
assert_(1 << 24)  # int32
assert_(1 << 48)  # int64
assert_(1 << 100)  # big

# comparisons
assert_(5 > 2)
assert_(2 + 1 == 3)
assert_(2 + 1 >= 3)
assert_(not (2 + 1 > 3))
assert_(2 + 2 <= 5)
assert_(not (2 + 1 < 3))
big = 1 << 100
assert_(big == big)
assert_(not (big != big))
assert_(big != big + 1)
assert_(not (big == big + 1))
assert_(big - 1 < big and big < big + 1)
assert_(-big - 1 < -big and -big < -big + 1)

# multiplication
assert_eq(1111 * 1111, 1234321)
assert_eq(1111 * 1, 1111)
assert_eq(1111 * -1, -1111)
assert_eq(1111 * 0, 0)
p1, p2 = 0x316c5239, 0x67c4a7d5  # 32-bit primes
product = p1 * p2
assert_eq(product, p2 * p1)
assert_eq(product // p1, p2)
assert_eq(product % p1, 0)
assert_eq(maxint, 0x7fffffff)
assert_eq(maxint * maxint, 0x3fffffff00000001)
assert_eq((1 << 62) - (2 << 31) + 1, 0x3fffffff00000001)
assert_eq(111111111 * 111111111, 12345678987654321)
assert_eq(-(111111111 * 111111111), -12345678987654321)
assert_eq((111111111 * 111111111) // 111111111, 111111111)

# 7 * 1317624576693539401 = maxlong
special_ints = [
    0,
    1,
    2,
    7,
    maxint - 1,
    maxint,
    maxint + 1,
    maxint + 2,
    0xffffffff,
    0x1ffffffff,
    1317624576693539401,
    maxlong - 1,
    maxlong,
    maxlong + 1,
    maxlong + 2,
]

def test_mul():
    for i_abs in special_ints:
        for j_abs in special_ints:
            for i_sign in [1, -1]:
                for j_sign in [1, -1]:
                    i = i_abs if i_sign > 0 else -i_abs
                    j = j_abs if j_sign > 0 else -j_abs
                    assert_eq(i * j, int_mul_slow(i, j))

test_mul()

# floored division
assert_eq(100 // 7, 14)
assert_eq(100 // -7, -15)
assert_eq(-100 // 7, -15)  # NB: different from Go / Java
assert_eq(-100 // -7, 14)  # NB: different from Go / Java
assert_eq(98 // 7, 14)
assert_eq(98 // -7, -14)
assert_eq(-98 // 7, -14)
assert_eq(-98 // -7, 14)
assert_eq(1 // 7, 0)
assert_eq(1 // -7, -1)
assert_eq(-1 // 7, -1)
assert_eq(-1 // -7, 0)
assert_eq(0 // 3, 0)
assert_eq(0 // -3, 0)
assert_eq(product // 1234567, 1169282890553)
assert_eq(product // -1234567, -1169282890553 - 1)
assert_eq(-product // 1234567, -1169282890553 - 1)
assert_eq(-product // -1234567, 1169282890553)
assert_eq(((-1) << 31) // -1, 1 << 31)  # sole case of int // int that causes int overflow
assert_eq(((-1) << 63) // -1, 1 << 63)  # ditto, long overflow
assert_fails(lambda: 1 // 0, "integer division by zero")

# floating-point division of int operands
assert_eq(str(100 / 7), "14.285714285714286")
assert_eq(str(100 / -7), "-14.285714285714286")
assert_eq(str(-100 / 7), "-14.285714285714286")
assert_eq(str(-100 / -7), "14.285714285714286")
assert_eq(type(98 / 7), "float")
assert_eq(98 / 7, 14.0)
assert_eq(98 / -7, -14.0)
assert_eq(-98 / 7, -14.0)
assert_eq(-98 / -7, 14.0)
assert_eq(type(product / 1234567), "float")
assert_eq(int(product / 1234567), 1169282890553)
assert_eq(int(product / -1234567), -1169282890553)
assert_eq(int(-product / 1234567), -1169282890553)
assert_eq(int(-product / -1234567), 1169282890553)
assert_eq(((-1) << 31) / -1, 1 << 31)  # sole case of int / int that causes int overflow
assert_eq(((-1) << 63) / -1, 1 << 63)  # ditto, long overflow

# remainder
assert_eq(100 % 7, 2)
assert_eq(100 % -7, -5)  # NB: different from Go / Java
assert_eq(-100 % 7, 5)  # NB: different from Go / Java
assert_eq(-100 % -7, -2)
assert_eq(98 % 7, 0)
assert_eq(98 % -7, 0)
assert_eq(-98 % 7, 0)
assert_eq(-98 % -7, 0)
assert_eq(product % 1234567, 1013598)
assert_eq(product % -1234567, -220969)  # ditto
assert_eq(-product % 1234567, 220969)  # ditto
assert_eq(-product % -1234567, -1013598)
assert_eq(1 % maxlong, 1)
assert_eq((-1) % maxlong, maxlong - 1)
assert_eq(1 % minlong, -maxlong)
assert_eq((-1) % minlong, -1)
assert_fails(lambda: 1 % 0, "integer modulo by zero")

# precedence
assert_eq(5 - 7 * 2 + 3, -6)
assert_eq(4 * 5 // 2 + 5 // 2 * 4, 18)
assert_eq(1 << 8 - 1, 1 << (8 - 1))  # confusingly...
assert_eq(8 | 3 ^ 4 & -2, 15)
assert_eq(~8 >> 1 | 3 ^ 4 & -2 << 2 * 3 + 4 // -2, -5)

# compound assignment
def compound():
    x = 1
    x += 1
    assert_eq(x, 2)
    x -= 3
    assert_eq(x, -1)
    x *= 10
    assert_eq(x, -10)
    x //= -2
    assert_eq(x, 5)
    x %= 3
    assert_eq(x, 2)

compound()

# unary operators

assert_eq(+0, 0)
assert_eq(+4, 4)
assert_eq(+(-4), -4)

assert_eq(-0, 0)
assert_eq(-4, 0 - 4)
assert_eq(-(0 - 4), 4)
assert_eq(-minint, 0 - minint)
assert_eq(-maxint, 0 - maxint)
assert_eq(-minlong, 0 - minlong)
assert_eq(-maxlong, 0 - maxlong)

assert_eq(+(+4), 4)
assert_eq(+(-4), 0 - 4)
assert_eq(-(+(-4)), 4)

# bitwise

def f():
    x = 2
    x &= 1
    assert_eq(x, 0)
    x = 0
    x |= 2
    assert_eq(x, 2)
    x ^= 3
    assert_eq(x, 1)
    x <<= 2
    assert_eq(x, 4)
    x >>= 2
    assert_eq(x, 1)

f()

assert_eq(minint & -1, minint)
assert_eq(maxint & -1, maxint)
assert_eq(minlong & -1, minlong)
assert_eq(maxlong & -1, maxlong)

assert_eq(minint | -1, -1)
assert_eq(maxint | -1, -1)
assert_eq(minlong | -1, -1)
assert_eq(maxlong | -1, -1)

assert_eq(minint ^ -1, maxint)
assert_eq(maxint ^ -1, minint)
assert_eq(minlong ^ -1, maxlong)
assert_eq(maxlong ^ -1, minlong)

assert_eq(~minint, maxint)
assert_eq(~maxint, minint)
assert_eq(~minlong, maxlong)
assert_eq(~maxlong, minlong)
assert_eq(~(1 << 100), -(1 << 100) - 1)
assert_eq(~(-(1 << 100)), (1 << 100) - 1)

# |
assert_eq(1 | 2, 3)
assert_eq(3 | 6, 7)
assert_eq(7 | 0, 7)

# &
assert_eq(7 & 0, 0)
assert_eq(7 & 7, 7)
assert_eq(7 & 2, 2)
assert_eq((1 | 2) & (2 | 4), 2)
assert_fails(lambda: 1 & False, "unsupported binary operation: int & bool")

# ^
assert_eq(1 ^ 2, 3)
assert_eq(2 ^ 2, 0)
assert_eq(-6 ^ 0, -6)
assert_eq(1 | 0 ^ 1, 1)  # check | and ^ operators precedence
assert_fails(lambda: "a" ^ 5, "unsupported binary operation: string \\^ int")

# ~
assert_eq(~1, -2)
assert_eq(~(-2), 1)
assert_eq(~0, -1)
assert_eq(~6, -7)
assert_eq(~0, -1)
assert_eq(~2147483647, -2147483647 - 1)
assert_fails(lambda: ~False, "unsupported unary operation: ~bool")

# <<
assert_eq(1 << 2, 4)
assert_eq(7 << 0, 7)
assert_eq(-1 << 31, -2147483647 - 1)
assert_eq(1 << 31, maxint + 1)
assert_eq(1 << 32, (maxint + 1) * 2)
assert_eq(1 << 63, maxlong + 1)
assert_eq(1 << 64, (maxlong + 1) * 2)
assert_eq(-1 << 31, minint)
assert_eq(-1 << 32, minint * 2)
assert_eq(-1 << 63, minlong)
assert_eq(-1 << 64, minlong * 2)
assert_fails(lambda: 1 << 520, "shift count too large: 520")
assert_fails(lambda: 1 << -4, "negative shift count: -4")

# >>
assert_eq(2 >> 1, 1)
assert_eq(7 >> 0, 7)
assert_eq(0 >> 0, 0)
assert_eq(minint >> 9999, -1)
assert_eq(minlong >> 9999, -1)
assert_eq(maxint >> 9999, 0)
assert_eq(maxlong >> 9999, 0)
assert_eq(minint >> 31, -1)
assert_eq(minint >> 30, -2)
assert_eq(minlong >> 63, -1)
assert_eq(minlong >> 62, -2)
assert_eq(maxint >> 31, 0)
assert_eq(maxint >> 30, 1)
assert_eq(maxlong >> 63, 0)
assert_eq(maxlong >> 62, 1)
assert_eq(1000 >> 100, 0)
assert_eq(-10 >> 1000, -1)
assert_fails(lambda: 2 >> -1, "negative shift count: -1")

# << and >>
assert_eq(1 << 500 >> 499, 2)
assert_eq(1 << 32, 0x10000 * 0x10000)
assert_eq(1 << 64, 0x10000 * 0x10000 * 0x10000 * 0x10000)
assert_eq(((0x1010 << 100) | (0x1100 << 100)) >> 100, 0x1110)
assert_eq(((0x1010 << 100) ^ (0x1100 << 100)) >> 100, 0x0110)
assert_eq(((0x1010 << 100) & (0x1100 << 100)) >> 100, 0x1000)
assert_eq(~((~(0x1010 << 100)) >> 100), 0x1010)
