---
layout: posts
title: Configuring your Java builds
---

Let say that you want to build for Java 8 and errorprone checks off but
keep the tools directory provided with Bazel in the package path, you could do
that by having the following rc file:

```
build --javacopt="-extra_checks:off"
build --javacopt="-source 8"
build --javacopt="-target 8"
```

However, the file would becomes quickly overloaded, especially if you take
all languages and options into account. Instead, you can tweak the
[java_toolchain](https://github.com/bazelbuild/bazel/tree/0e1680e58f01f3d443f7e68865b5a56b76c9dadf/tools/jdk/BUILD#L73)
rule that specifies the various options for the java compiler. So in a
BUILD file:

```python
java_toolchain(
    name = "my_toolchain",
    encoding = "UTF-8",
    source_version = "8",
    target_version = "8",
    misc = [
        "-extra_checks:on",
    ],
)
```

And to keep it out of the tools directory (or you need to copy the rest
of the package), you can redirect the default one in a bazelrc:

```
build --java_toolchain=//package:my_toolchain
```

In the future, toolchain rules should be the configuration points for all
the languages but it is a long road. We also want to make it easier to
rebind the toolchain using the `bind` rule in the WORKSPACE file.

