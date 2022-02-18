---
layout: documentation
title: Variable x is read only
---

<div style="background-color: #EFCBCB; color: #AE2B2B;  border: 1px solid #AE2B2B; border-radius: 5px; border-left: 10px solid #AE2B2B; padding: 0.5em;">
<b>IMPORTANT:</b> The Bazel docs have moved! Please update your bookmark to <a href="https://bazel.build/rules/errors/read-only-variable" style="color: #0000EE;">https://bazel.build/rules/errors/read-only-variable</a>
<p/>
You can <a href="https://blog.bazel.build/2022/02/17/Launching-new-Bazel-site.html" style="color: #0000EE;">read about</a> the migration, and let us <a href="https://forms.gle/onkAkr2ZwBmcbWXj7" style="color: #0000EE;">know what you think</a>.
</div>


# Error: Variable x is read only

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
