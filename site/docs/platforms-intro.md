---
layout: documentation
title: Building With Platforms
---

# Building With Platforms

- [Overview](#overview)
- [Background](#background)
- [Goal](#goal)

## Overview

Bazel has sophisticated support for modeling [platforms](platforms.html) and
[toolchains](toolchains.html). Integrating this into real projects requires
coherent cooperation between project and library owners, rule maintainers,
and core Bazel devs.

This page summarizes the arguments for using platforms and shows how to
navigate these relationships for maximum value with minimum cognitive
overhead.

For more formal documentation, see:

* [Platforms](platforms.html)
* [Toolchains](toolchains.html)

## Background

### The Problem

*Platforms* and *toolchains* were introduced to *standardize* the need for
 software projects to target different kinds of computers with different
 language-appropriate tools.

This is a relatively recent addition to Bazel. It was
[inspired](https://blog.bazel.build/2019/02/11/configurable-builds-part-1.html)
by the observation that language maintainers were *already* doing this in ad hoc
and incompatible ways. For example, C++ rules use `--cpu` and `--crosstool_top`
to set a build's target CPU and C++ toolchain. Neither of these represents a
complete "platform". Historic attempts to use them for that inevitably led to
awkward and inaccurate build APIs. They also don't say anything about Java
toolchains, which evolved their own independent interface with
`--java_toolchain`.

Bazel aims to excel at large, mixed-language, multi-platform projects. This
demands more principled support for these concepts, including clear APIs that
bind rather than diverge languages and projects. This is what the new platform
and toolchain APIs achieve.

### Migration

But this isn't enough for all projects to use platforms. It's also necessary to
stop using the old APIs. This isn't trivial because all of a project's
languages, toolchains, dependencies, and `select()`s have to support the new
APIs. This requires an *ordered migration sequence* if you don't want your
project to break.

For example, Bazel's [C++](/versions/master/bazel-and-cpp.html) rules aleady
support platforms while the [Android](/versions/master/bazel-and-android.html)
rules don't. *Your* C++ project may not care about Android. But others may. So
it's not yet safe to globally enable platform support for all C++ builds.

The thrust of this page describes this migration sequence and how and when your
projects can fit in.

## Goal

Bazel's platform migration is complete when all projects can build with the form:

```sh
$ bazel build //myproject --platforms=//my:platform
```

This implies






