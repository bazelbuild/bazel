---
layout: posts
title: Using Skylark remote repositories to auto-detect the C++ toolchain.
---

[Skylark remote repositories](/docs/skylark/repository_rules.html) let you
create custom [external repositories](/docs/external.html) using
[Skylark](/docs/skylark/index.html). This not only enables creating rules for
custom package systems such as [PyPi](https://pypi.python.org) but also generating
a repository to reflect the toolchain installed on the workstation Bazel is running
on. We explain here how we implemented [auto-configuration for the C++
toolchain](https://github.com/bazelbuild/bazel/blob/master/tools/cpp/cc_configure.bzl).

## Principles

<blockquote>
C++ toolchain: the set of binaries and libraries required to build C++ code.

Crosstool: a compiler capable of building for a certain architecture, which
can be different from the host architecture (e.g., gcc running on Linux and
building for Raspberry Pi).
</blockquote>

C++ toolchains are configured in Bazel using a [crosstool target](https://github.com/bazelbuild/bazel/blob/8fa5ae6a6364100f2a7f9130e62eb0edb447339a/tools/cpp/BUILD#L32)
and a [CROSSTOOL file](https://github.com/bazelbuild/bazel/blob/master/tools/cpp/CROSSTOOL).

This crosstool target (:default_toolchain) is the first step in moving the contents
of the CROSSTOOL file entirely into BUILD file rules. The CROSSTOOL file defines
where to find the C++ compiler, its include directories and also the various flag
to use at each compilation step.

When your C++ compiler is not in the standard location, then this static
CROSSTOOL file cannot find it. To cope with the variety of installation out
there, we created a `cc_configure` Skylark repository rule that will generates
a `@local_config_cc//tools/cpp` package containing a generated CROSSTOOL file
based on the information we gathered from the operating system.


## Implementation

The [`cc_configure`](https://github.com/bazelbuild/bazel/blob/9116b3e99af2fd31d92c9bb7c37905a1675456c1/tools/cpp/cc_configure.bzl#L291)
rule is actually a macro wrapping the [`cc_autoconf`](https://github.com/bazelbuild/bazel/blob/9116b3e99af2fd31d92c9bb7c37905a1675456c1/tools/cpp/cc_configure.bzl#L288)
enforcing the `local_config_cc` name for the repository. The
[implementation](https://github.com/bazelbuild/bazel/blob/9116b3e99af2fd31d92c9bb7c37905a1675456c1/tools/cpp/cc_configure.bzl#L255)
of the `cc_autoconf` rule does the following step:

 - [Detect the `cpu_value`](https://github.com/bazelbuild/bazel/blob/9116b3e99af2fd31d92c9bb7c37905a1675456c1/tools/cpp/cc_configure.bzl#L85)
   using the [`repository_ctx.os.name`](/docs/skylark/lib/repository_os.html#name) attribute.
 - Generates a [static package](https://github.com/bazelbuild/bazel/blob/9116b3e99af2fd31d92c9bb7c37905a1675456c1/tools/cpp/cc_configure.bzl#L85)
   if we do not support the target platform.
 - Detect the [C++ compiler path](https://github.com/bazelbuild/bazel/blob/9116b3e99af2fd31d92c9bb7c37905a1675456c1/tools/cpp/cc_configure.bzl#L235)
   using [`repository_ctx.which`](/docs/skylark/lib/repository_ctx.html#which) and the `CC` environment variable with
   [`repository_ctx.os.environ`](/docs/skylark/lib/repository_os.html#environ).
 - Detect some [more tool paths](https://github.com/bazelbuild/bazel/blob/9116b3e99af2fd31d92c9bb7c37905a1675456c1/tools/cpp/cc_configure.bzl#L53),
   still using [`repository_ctx.which`](/docs/skylark/lib/repository_ctx.html#which).
 - Generates the [various flag for the `CROSSTOOL` file](https://github.com/bazelbuild/bazel/blob/9116b3e99af2fd31d92c9bb7c37905a1675456c1/tools/cpp/cc_configure.bzl#L127),
   [testing flags against the detected compiler](https://github.com/bazelbuild/bazel/blob/9116b3e99af2fd31d92c9bb7c37905a1675456c1/tools/cpp/cc_configure.bzl#L114)
   using [`repository_ctx.execute`](/docs/skylark/lib/repository_ctx.html#execute). We also
   [detect the include directories](https://github.com/bazelbuild/bazel/blob/9116b3e99af2fd31d92c9bb7c37905a1675456c1/tools/cpp/cc_configure.bzl#L101)
   with [`repository_ctx.execute`](/docs/skylark/lib/repository_ctx.html#execute).
 - With the gathered information, generate the C++ tools package: its [BUILD file](https://github.com/bazelbuild/bazel/blob/9116b3e99af2fd31d92c9bb7c37905a1675456c1/tools/cpp/cc_configure.bzl#L274),
   [wrapper script for Darwin](https://github.com/bazelbuild/bazel/blob/9116b3e99af2fd31d92c9bb7c37905a1675456c1/tools/cpp/cc_configure.bzl#L278) and
   [CROSSTOOL file](https://github.com/bazelbuild/bazel/blob/9116b3e99af2fd31d92c9bb7c37905a1675456c1/tools/cpp/cc_configure.bzl#L279) using
   [`repository_ctx.template`](/docs/skylark/lib/repository_ctx.html#template).

So using the function provided by [`repository_ctx`](/docs/skylark/lib/repository_ctx.html), we can discover
the binaries on the system, what version they are, and which options they support, then generate a
configuration to match the local C++ toolchain.


## Creating your own repository rules

When creating a Skylark remote repository, a few things should be taken in considerations:

 - The Skylark implementation of a remote repository is run during the loading phase of
   the repository, which means that unless the rule definition is changed in the WORKSPACE
   file or the implementation fails, it will not be re-run unless the user does a
   `bazel clean --expunge`. We are thinking of further command to force re-run that loading
   phase for a specific remote repository ([#974](https://github.com/bazelbuild/bazel/issues/974)).
 - Skylark remote repository can do a lot of non hermetic operation, it is recommended
   to check as many things as possible to ensure hermeticity (and overall, we recommend
   using a vendored toolchain instead of using auto-detected one if reproducibility is important).
   For example, it is recommended to use the `sha256` argument of the
   [`repository_ctx.download`](/docs/skylark/lib/repository_ctx.html#download) method.
 - Naming a rule can be complex and we recommend to not use standard suffix of classical
   rules for remote repositories (e.g. `*_library` or `*_binary`). If you create a
   package rule, a good name would probably be `xxx_package` (e.g., `pypi_package`). If
   you create an autoconfiguration rule, `xxx_configure` is probably the best name
   (e.g. `cc_configure`).
