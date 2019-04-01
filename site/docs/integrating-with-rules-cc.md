---
layout: documentation
title: Integrating with C++ rules
---

# Integrating with C++ rules

This page describes how to integrate with C++ rules on various levels.

# Accessing the C++ toolchain

Because of
[ongoing migration of C++ rules](https://github.com/bazelbuild/bazel/issues/6516)
to [platforms](https://docs.bazel.build/versions/master/platforms.html) and
[toolchains](https://docs.bazel.build/versions/master/toolchains.html), we
advise to use the helper function available at
[@bazel_tools//tools/cpp:toolchain_utils.bzl](https://source.bazel.build/bazel/+/master:tools/cpp/toolchain_utils.bzl;l=23),
which works both when toolchains are disabled and enabled. To depend on a C++
toolchain in your rule, add a
[`Label`](https://docs.bazel.build/versions/master/skylark/lib/attr.html#label)
attribute named `_cc_toolchain` and point it
to `@bazel_tools//tools/cpp:current_cc_toolchain` (an instance of
`cc_toolchain_alias` rule, that points to the currently selected C++ toolchain).
Then, in the rule implementation, use
[`find_cpp_toolchain(ctx)`](https://source.bazel.build/bazel/+/master:tools/cpp/toolchain_utils.bzl;l=23)
to get the
[`CcToolchainInfo`](https://docs.bazel.build/versions/master/skylark/lib/CcToolchainInfo.html).
A complete working example can be found
[in the rules_cc examples](https://github.com/bazelbuild/rules_cc/blob/master/examples/write_cc_toolchain_cpu/write_cc_toolchain_cpu.bzl).

# Generating command lines and environment variables using the C++ toolchain

Typically, you would integrate with the C++ toolchain to have the same
command line flags as C++ rules do, but without using C++ actions directly.
This is because when writing our own actions, they must behave
consistently with the C++ toolchain - for example, passing C++ command line
flags to a tool that invokes the C++ compiler behind the scenes.

C++ rules use a special way of constructing command lines based on [feature
configuration](cc-toolchain-config-reference.html). To construct a command line,
you need the following:

* `features` and `action_configs` - these come from the `CcToolchainConfigInfo`
  and encapsulated in `CcToolchainInfo`
* `FeatureConfiguration` - returned by [cc_common.configure_features](https://docs.bazel.build/versions/master/skylark/lib/cc_common.html#configure_features)
* cc toolchain config variables - returned by
  [cc_common.create_compile_variables](https://docs.bazel.build/versions/master/skylark/lib/cc_common.html#create_compile_variables)
  or
  [cc_common.create_link_variables](https://docs.bazel.build/versions/master/skylark/lib/cc_common.html#create_link_variables).

There still are tool-specific getters, such as
[compiler_executable](https://docs.bazel.build/versions/master/skylark/lib/CcToolchainInfo.html#compiler_executable).
Prefer `get_tool_for_action` over these, as tool-specific getters will
eventually be removed.

A complete working example can be found
[in the rules_cc examples](https://github.com/bazelbuild/rules_cc/blob/master/examples/my_c_compile/my_c_compile.bzl).

# Implementing Starlark rules that depend on C++ rules and/or that C++ rules can depend on

Most C++ rules provide
[`CcInfo`](https://docs.bazel.build/versions/master/skylark/lib/CcInfo.html),
a provider containing [`CompilationContext`](https://docs.bazel.build/versions/master/skylark/lib/CompilationContext.html)
and
[`LinkingContext`](https://docs.bazel.build/versions/master/skylark/lib/LinkingContext.html).
Through these it is possible to access information such as all transitive headers
or libraries to link. From `CcInfo` and from the `CcToolchainInfo` custom
Starlark rules should be able to get all the information they need.

If a custom Starlark rule provides `CcInfo`, it's a signal to the C++ rules that
they can also depend on it. Be careful, however - if you only need to propagate
`CcInfo` through the graph to the binary rule that then makes use of it, wrap
`CcInfo` in a different provider. For example, if `java_library` rule wanted
to propagate native dependencies up to the `java_binary`, it shouldn't provide
`CcInfo` directly (`cc_binary` depending on `java_library` doesn't make sense),
it should wrap it in, for example, `JavaCcInfo`.

A complete working example can be found
[in the rules_cc examples](https://github.com/bazelbuild/rules_cc/blob/master/examples/my_c_archive/my_c_archive.bzl).


# Reusing logic and actions of C++ rules

_Not stable yet, we will update this section once the API stabilizes. Follow
[#4570](https://github.com/bazelbuild/bazel/issues/4570) for up to date
information._
