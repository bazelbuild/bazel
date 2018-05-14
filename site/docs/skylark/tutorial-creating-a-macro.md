---
layout: documentation
title: Creating a Macro
---

# Creating a Macro

Let's suppose you need to run a tool as part of your build. For example, you
may want to generate or preprocess a source file, or compress a binary. In this
tutorial, we are going to resize an image.

The easiest way is to use a `genrule`:

``` python
genrule(
    name = "logo_miniature",
    srcs = ["logo.png"],
    outs = ["small_logo.png"],
    cmd = "convert $< -resize 100x100 $@",
)

cc_binary(
    name = "my_app",
    srcs = ["my_app.cc"],
    data = [":logo_miniature"],
)
```

If you need to resize more images, you may want to reuse the code. To do that,
we are going to define a function in a separate `.bzl` file. Let's call the file
`miniature.bzl`:

``` python
def miniature(name, src, size="100x100", **kwargs):
  """Create a miniature of the src image.

  The generated file is prefixed with 'small_'.
  """
  native.genrule(
    name = name,
    srcs = [src],
    outs = ["small_" + src],
    cmd = "convert $< -resize 100x100 $@",
    **kwargs
  )
```

A few remarks:

* By convention, macros have a `name` argument, just like rules.

* We document the behavior of a macro by using a
  [docstring](https://www.python.org/dev/peps/pep-0257/) like in Python.

* To call a `genrule`, or any other native rule, use `native.`.

* `**kwargs` is used to forward the extra arguments to the underlying `genrule`
  (it works just like in [Python](https://docs.python.org/3/tutorial/controlflow.html#keyword-arguments)).
  This is useful, so that a user can use standard attributes like `visibility`,
  or `tags`.

Now, you can use the macro from the `BUILD` file:

``` python
load("//path/to:miniature.bzl", "miniature")

miniature(
    name = "logo_miniature",
    src = "image.png",
)

cc_binary(
    name = "my_app",
    srcs = ["my_app.cc"],
    data = [":logo_miniature"],
)
```

Macros are suitable for simple tasks. If you want to do anything more
complicated, for example add support for a new programming language, consider
creating a [rule](rules.md). Rules will give you more control and flexibility.
