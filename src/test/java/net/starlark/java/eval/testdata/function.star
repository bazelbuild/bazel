# 2-step function call

x = [1, 2, 3]
y = x.clear
assert_eq(x, [1, 2, 3])
z = y()
assert_eq(z, None)
assert_eq(x, [])
---

x = {1: 2}
y = x.pop
assert_eq(x, {1: 2})
z = y(1)
assert_eq(z, 2)
assert_eq(x, {})
---

x = "hello"
y = x.upper
z = y()
assert_eq(z, "HELLO")
assert_eq(x, "hello")
---

x = "abc"
y = x.index
z = y("b")
assert_eq(z, 1)
assert_eq(x, "abc")
---

y = {}
assert_eq(y.clear == y.clear, False)
assert_eq([].clear == [].clear, False)
assert_eq(type([].clear), "builtin_function_or_method")
assert_eq(str([].clear), "<built-in method clear of list value>")
assert_eq(str({}.clear), "<built-in method clear of dict value>")
assert_eq(str(len), "<built-in function len>")

assert_fails({}.pop, "missing 1 required positional argument: key")

# Arguments are evaluated in left-to-right order.
# See https://github.com/bazelbuild/starlark/issues/13.
order = []

def id(x):
  order.append(x)
  return x

def f(*args, **kwargs):
  return args, kwargs

assert_eq(
  f(id(1), id(2), x=id(3), *[id(4)], **dict(z=id(5))),
  ((1, 2, 4), {"x": 3, "z": 5}))
assert_eq(order, [1, 2, 3, 4, 5])

---
# getattr

assert_eq(getattr("abc", "upper")(), "ABC")
assert_eq(getattr({'a': True}, "pop")('a'), True)
assert_eq(getattr({}, "hello", "default"), "default")

y = [1, 2, 3]
x = getattr(y, "clear")
assert_eq(y, [1, 2, 3])
x()
assert_eq(y, [])

assert_fails(lambda: getattr("", "abc"), "'string' value has no field or method 'abc'")

assert_fails(lambda: getattr("", "pop", "clear")(), "'string' object is not callable")

# Regression test for a type mismatch crash (b/168743413).
assert_fails(lambda: getattr(1, []), "parameter 'name' got value of type 'list', want 'string'")

# assert_fails evaluates an expression (passed unevaluated in the form of
# a lambda) and asserts that evaluation fails with the given error.
assert_fails(lambda: 1//0, 'integer division by zero')

---
# Test of nested def statements.
def adder(x):
  def add(x, y): return x + y # no free vars
  def adder(y): return add(x, y) # freevars={x, add}
  return adder

add3 = adder(3)
assert_eq(add3(1), 4)
assert_eq(add3(-1), 2)

addlam = adder("lam")
assert_eq(addlam("bda"), "lambda")
assert_eq(addlam("bada"), "lambada")


# Same, with lambda
def adder2(x):
  return lambda y: x+y

assert_eq(adder2(3)(1), 4)
assert_eq(adder2("lam")("bda"), "lambda")


# Test of stateful function values.
def makerand(seed=0):
  "makerand returns a stateful generator of small pseudorandom numbers."
  state = [seed]
  def rand():
    "rand returns the next pseudorandom number in the sequence."
    state[0] = ((state[0] + 7207) * 9941) & 0xfff
    return state[0]
  return rand

rand1 = makerand(123)
rand2 = makerand(123)
assert_eq([rand1() for _ in range(10)], [3786, 133, 796, 1215, 862, 1961, 3088, 4035, 1458, 3981])
assert_eq([rand2() for _ in range(10)], [3786, 133, 796, 1215, 862, 1961, 3088, 4035, 1458, 3981])

# different seed
rand3 = makerand()
assert_eq([rand3() for _ in range(10)], [1651, 1570, 3261, 3508, 1335, 1846, 2657, 3880, 699, 3594])

# Attempt to mutate frozen closure state.
freeze()
assert_fails(rand3, "trying to mutate a frozen list value")

---
# recursion is disallowed
def fib(x):
    return x if x < 2 else fib(x-1)+fib(x-2)

assert_fails(lambda: fib(10), "function 'fib' called recursively")

---
# The recursion check breaks function encapsulation:
# A function g that internally uses a higher-order helper function
# such as 'call' (or Python's map and reduce) cannot itself be
# called from within an active call of that helper.
def call(f): f()
def g(): call(list)
assert_fails(lambda: call(g), "function 'call' called recursively")

---
# The recursion check is based on the syntactic equality
# (same def statement), not function value equivalence.
def eta(f):
    return lambda: f()

def nop(): pass

# fn1 and fn2 are both created by the same lambda,
# but they are distinct and close over different values...
fn1 = eta(nop)
fn2 = eta(fn1)
assert_eq(str(fn1), '<function lambda>')
assert_eq(str(fn2), '<function lambda>')
assert_(fn1 != fn2)

# ...yet both cannot be called in the same thread:
assert_fails(fn2, "function 'lambda' called recursively")

# This rule prevents users from writing the Y combinator,
# which creates a new closure at each step of the recursion.
Y = lambda f: (lambda x: x(x))(lambda y: f(lambda *args: y(y)(*args)))
fibgen = lambda fib: lambda x: (x if x<2 else fib(x-1)+fib(x-2))
fib2 = Y(fibgen)
assert_fails(lambda: [fib2(x) for x in range(10)], "function 'lambda' called recursively")

---
# Trivial test of lambda.
def map(f, list):
    return [f(x) for x in list]

assert_eq(map(lambda x: len(x), ["one", "two", "three"]),
          [3, 3, 5])

assert_eq(type(lambda: 0), "function")
assert_eq(str(lambda: 0), "<function lambda>")

---
# builder returns a string builder:
# an opaque, stateful value with methods and open recursion.
def builder():
    chunks = []
    self = None
    def append(x):
        chunks.append("%s" % x)
        return self
    def build():
        return "".join(chunks)
    self = struct(append = append, build = build)
    return self

assert_eq(builder().append(1).append(" + ").append(2).build(), "1 + 2")
