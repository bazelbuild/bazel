# for statements and list comprehensions

# --- for statements ---

# sequence assignment

# Assignment to empty list/tuple is permitted.
# https://github.com/bazelbuild/starlark/issues/93 for discussion.
() = ()
[] = ()

# RHS not iterable
x, y = 1  ### got 'int' in sequence assignment
---
(x,) = 1 ### got 'int' in sequence assignment
---
[x] = 1 ### got 'int' in sequence assignment
---

# too few
x, y = () ### too few values to unpack \(got 0, want 2\)
---
[x, y] = () ### too few values to unpack \(got 0, want 2\)
---

# just right
x, y = 1, 2
---
[x, y] = 1, 2
---
(x,) = [1]
---

# too many
() = 1 ### got 'int' in sequence assignment
---
() = (1,) ### too many values to unpack \(got 1, want 0\)
---
x, y = 1, 2, 3 ### too many values to unpack \(got 3, want 2\)
---
[x, y] = 1, 2, 3 ### too many values to unpack \(got 3, want 2\)
---

# Assignment to empty tuple is permitted.
# See https://github.com/bazelbuild/starlark/issues/93 for discussion.
assert_eq([1 for [] in [(), []]], [1, 1])

# Iterating over dict without .items() gives informative error.
assert_eq([v for v in dict(a = "b")], ["a"])
[None for () in dict(a = "b")]  ### got 'string' in sequence assignment \(want 0-element sequence\)
---
[None for (v1,) in dict(a = "b")] ### got 'string' in sequence assignment \(want 1-element sequence\)
---
[None for v1, v2 in dict(a = "b")] ### got 'string' in sequence assignment \(want 2-element sequence\)
---

# --- list comprehensions ---

assert_eq(["foo/%s.java" % x for x in []], [])
assert_eq(
    ["foo/%s.java" % y for y in ["bar", "wiz", "quux"]],
    ["foo/bar.java", "foo/wiz.java", "foo/quux.java"])
assert_eq(
    ["%s/%s.java" % (z, t)
     for z in ["foo", "bar"]
     for t in ["baz", "wiz", "quux"]],
    ["foo/baz.java",
     "foo/wiz.java",
     "foo/quux.java",
     "bar/baz.java",
     "bar/wiz.java",
     "bar/quux.java"])
assert_eq(
    ["%s/%s.java" % (b, b)
     for a in ["foo", "bar"]
     for b in ["baz", "wiz", "quux"]],
    ["baz/baz.java",
     "wiz/wiz.java",
     "quux/quux.java",
     "baz/baz.java",
     "wiz/wiz.java",
     "quux/quux.java"])
assert_eq(
    ["%s/%s.%s" % (c, d, e)
     for c in ["foo", "bar"]
     for d in ["baz", "wiz", "quux"]
     for e in ["java", "cc"]],
    ["foo/baz.java",
     "foo/baz.cc",
     "foo/wiz.java",
     "foo/wiz.cc",
     "foo/quux.java",
     "foo/quux.cc",
     "bar/baz.java",
     "bar/baz.cc",
     "bar/wiz.java",
     "bar/wiz.cc",
     "bar/quux.java",
     "bar/quux.cc"])
assert_eq([i for i in (1, 2)], [1,2])
assert_eq([i for i in [2, 3] or [1, 2]], [2, 3])

# nested list comprehensions
li = [[1, 2], [3, 4]]
assert_eq([j for i in li for j in i], [1,2,3,4])
input = [["abc"], ["def", "ghi"]]
assert_eq(
    ["%s %s" % (b, c)
     for a in input
     for b in a
     for c in b.elems()],
    ["abc a", "abc b", "abc c", "def d", "def e", "def f", "ghi g", "ghi h", "ghi i"])

# filtering
range3 = [0, 1, 2]
assert_eq([a for a in (4, None, 2, None, 1)
           if a != None],
          [4, 2, 1])
assert_eq([b+c for b in [0, 1, 2]
           for c in [0, 1, 2]
           if b + c > 2],
          [3, 3, 4])
assert_eq([d+e for d in range3
           if d % 2 == 1
           for e in range3],
          [1, 2, 3])
assert_eq([[f, g] for f in [0, 1, 2, 3, 4]
           if f
           for g in [5, 6, 7, 8]
           if f * g % 12 == 0],
          [[2, 6], [3, 8], [4, 6]])
assert_eq([h for h in [4, 2, 0, 1] if h], [4, 2, 1])

# multiple variables, ok
assert_eq([x + y for x, y in [(1, 2), (3, 4)]], [3, 7])
assert_eq([z + t for (z, t) in [[1, 2], [3, 4]]], [3, 7])

# multiple variables, fail
[x + y for x, y, z in [(1, 2), (3, 4)]] ### too few values to unpack \(got 2, want 3\)
---
[x + y for x, y in (1, 2)] ### got 'int' in sequence assignment \(want 2-element sequence\)
---
[x + y for x, y, z in [(1, 2), (3, 4)]] ### too few values to unpack \(got 2, want 3\)
---
[x + y for x, y in (1, 2)] ### got 'int' in sequence assignment \(want 2-element sequence\)
---
[x + y for x, y, z in [(1, 2), (3, 4)]] ### too few values to unpack \(got 2, want 3\)
---
[x2 + y2 for x2, y2 in (1, 2)] ### got 'int' in sequence assignment \(want 2-element sequence\)
---
