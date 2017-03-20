---
layout: posts
title: Skylark and Java rules interoperability
---

As of Bazel 0.4.4, Java compilation is possible from a Skylark rule. This
facilitates the Skylark and Java interoperability and allows creating what we
call _Java sandwiches_ in Bazel.

## What is a Bazel Java sandwich?

A Java sandwich refers to custom rules written in Skylark being able to depend
on Bazel native rules (e.g. `java_library`) and the other way around. A typical
Java sandwich in Bazel could be illustrated like this:

```python
java_library(name = "top", ...)
java_skylark_library(name = "middle", deps = [":top", ...], ...)
java_library(name = "bottom", deps = [":middle", ...], ...)
```
## Built-in support for Java

In Skylark, an interface to built-in Java functionality is available via the `java_common` module.
The full API can be found in [the documentation](https://bazel.build/versions/master/docs/skylark/lib/java_common.html).

### `java_common.compile`

Compiles Java source files/jars from the implementation of a Skylark rule and
returns a `java_common.provider` that encapsulates the compilation details.

### `java_common.merge`

Merges the given providers into a single `java_common.provider`.


## Examples

To allow other Java rules (native or custom) to depend on a Skylark rule, the
Skylark rule should return a `java_common.provider`. All native Java rules
return `java_common.provider` by default, which makes it possible for any Java
related Skylark rule to depend on them.

For now, there are 3 ways of creating a `java_common.provider`:

1. The result of `java_common.compile`.
2. Fetching it from a Java dependency.
3. Merging multiple `java_common.provider` instances using `java_common.merge`.

### Using the Java sandwich with compilation example

This example illustrates the typical Java sandwich described above, that will
make use of Java compilation:

```python
java_library(name = "top", ...)
java_skylark_library(name = "middle", deps = [":top", ...], ...)
java_library(name = "bottom", deps = [":middle", ...], ...)
```

In the BUILD file we load the Skylark rule and have the rules:

```python
load(':java_skylark_library.bzl', 'java_skylark_library')

java_library(
  name = "top",
  srcs = ["A.java"],
  deps = [":middle"]
)

java_skylark_library(
  name = "middle",
  srcs = ["B.java"],
  deps = [":bottom"]
)

java_library(
  name = "bottom",
  srcs = ["C.java"]
)
```

The implementation of `java_skylark_library` rule does the following:

1. Collects all the `java_common.provider`s from its dependencies and merges
them using `java_common.merge`.
2. Creates an artifact that will be the output jar of the Java compilation.
3. Compiles the specified Java source files using `java_common.compile`, passing
as dependencies the collected `java_common.provider`s.
4. Returns the output jar and the `java_common.provider` resulting from the
compilation.

```python
def _impl(ctx):
  deps = []
  for dep in ctx.attr.deps:
    if java_common.provider in dep:
      deps.append(dep[java_common.provider])

  output_jar = ctx.new_file("lib" + ctx.label.name + ".jar")

  compilation_provider = java_common.compile(
    ctx,
    source_files = ctx.files.srcs,
    output = output_jar,
    javac_opts = [],
    deps = deps,
    strict_deps = "ERROR",
    java_toolchain = ctx.attr._java_toolchain,
    host_javabase = ctx.attr._host_javabase
  )
  return struct(
    files = set([output_jar]),
    providers = [compilation_provider]
  )

java_skylark_library = rule(
  implementation = _impl,
  attrs = {
    "srcs": attr.label_list(allow_files=True),
    "deps": attr.label_list(),
    "_java_toolchain": attr.label(default = Label("@bazel_tools//tools/jdk:toolchain")),
    "_host_javabase": attr.label(default = Label("//tools/defaults:jdk"))
  },
  fragments = ["java"]
)
```

### Just passing around information about Java rules example

In some use cases there is no need for Java compilation, but rather just passing
information about Java rules around. A Skylark rule can have some other
(irrelevant here) purpose, but if it is placed somewhere between two Java rules
it should not lose information from bottom to top.

In this example we have the same Bazel sandwich as above:

```python
java_library(name = "top", ...)
java_skylark_library(name = "middle", deps = [":top", ...], ...)
java_library(name = "bottom", deps = [":middle", ...], ...)
```

only that `java_skylark_library` won't make use of Java compilation, but will
make sure that all the Java information encapsulated by the Java library
`bottom` will be passed on to the Java library `top`.

The BUILD file is identical to the one from the previous example.

The implementation of `java_skylark_library` rule does the following:

1. Collects all the `java_common.provider`s from its dependencies
2. Returns the `java_common.provider` that resulted from merging the collected
dependencies.

```python
def _impl(ctx):
  deps = []
  for dep in ctx.attr.deps:
    if java_common.provider in dep:
      deps.append(dep[java_common.provider])
  deps_provider = java_common.merge(deps)
  return struct(
    providers = [deps_provider]
  )

java_skylark_library = rule(
  implementation = _impl,
  attrs = {
    "srcs": attr.label_list(allow_files=True),
    "deps": attr.label_list(),
    "_java_toolchain": attr.label(default = Label("@bazel_tools//tools/jdk:toolchain")),
    "_host_javabase": attr.label(default = Label("//tools/defaults:jdk"))
  },
  fragments = ["java"]
)
```
## More to come

Right now there is no way of creating a `java_common.provider` that encapsulates
compiled code (and its transitive dependencies), other than
`java_common.compile`. For example one may want to create a provider from a
`.jar` file produced by some other means.

Soon there will be support for use cases like this. Stay tuned!

If you are interested in tracking the progress on Bazel Java sandwich you can
subscribe to [this Github issue](https://github.com/bazelbuild/bazel/issues/2614).

_[Irina Iancu](https://github.com/iirina), on behalf of the Bazel Java team_
