Project: /_project.yaml
Book: /_book.yaml

# Error: Variable x is read only

{% include "_buttons.html" %}

A global variable cannot be reassigned. It will always point to the same object.
However, its content might change, if the value is mutable (for example, the
content of a list). Local variables don't have this restriction.

```python
a = [1, 2]

a[1] = 3

b = 3

b = 4  # forbidden
```

`ERROR: /path/ext.bzl:7:1: Variable b is read only`

You will get a similar error if you try to redefine a function (function
overloading is not supported), for example:

```python
def foo(x): return x + 1

def foo(x, y): return x + y  # forbidden
```
