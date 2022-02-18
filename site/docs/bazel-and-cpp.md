---
layout: documentation
title: C++ and Bazel
---

<div style="background-color: #EFCBCB; color: #AE2B2B;  border: 1px solid #AE2B2B; border-radius: 5px; border-left: 10px solid #AE2B2B; padding: 0.5em;">
<b>IMPORTANT:</b> The Bazel docs have moved! Please update your bookmark to <a href="https://bazel.build/docs/bazel-and-cpp" style="color: #0000EE;">https://bazel.build/docs/bazel-and-cpp</a>
<p/>
You can <a href="https://blog.bazel.build/2022/02/17/Launching-new-Bazel-site.html" style="color: #0000EE;">read about</a> the migration, and let us <a href="https://forms.gle/onkAkr2ZwBmcbWXj7" style="color: #0000EE;">know what you think</a>.
</div>


# C++ and Bazel

This page contains resources that help you use Bazel with C++ projects. It links
to a tutorial, build rules, and other information specific to building C++
projects with Bazel.

## Working with Bazel

The following resources will help you work with Bazel on C++ projects:

*  [Tutorial: Building a C++ project](tutorial/cpp.html)
*  [C++ common use cases](cpp-use-cases.html)
*  [C/C++ rules](be/c-cpp.html)
*  [C++ toolchain configuration](cc-toolchain-config-reference.html)
*  [Tutorial: Configuring C++ toolchains](tutorial/cc-toolchain-config.html)
*  [Integrating with C++ rules](integrating-with-rules-cc.html)

## Best practices

In addition to [general Bazel best practices](best-practices.html), below are
best practices specific to C++ projects.

### BUILD files

Follow the guidelines below when creating your BUILD files:

*  Each `BUILD` file should contain one [`cc_library`](be/c-cpp.html#cc_library)
   rule target per compilation unit in the directory.

*  You should granularize your C++ libraries as much as
   possible to maximize incrementality and parallelize the build.

*  If there is a single source file in `srcs`, name the library the same as
   that C++ file's name. This library should contain C++ file(s), any matching
   header file(s), and the library's direct dependencies. For example:

   ```python
   cc_library(
       name = "mylib",
       srcs = ["mylib.cc"],
       hdrs = ["mylib.h"],
       deps = [":lower-level-lib"]
   )
   ```

*  Use one `cc_test` rule target per `cc_library` target in the file. Name the
   target `[library-name]_test` and the source file `[library-name]_test.cc`.
   For example, a test target for the `mylib` library target shown above would
   look like this:

   ```python
   cc_test(
       name = "mylib_test",
       srcs = ["mylib_test.cc"],
       deps = [":mylib"]
   )
   ```

### Include paths

Follow these guidelines for include paths:

*  Make all include paths relative to the workspace directory.

*  Use quoted includes (`#include "foo/bar/baz.h"`) for non-system headers, not
   angle-brackets (`#include <foo/bar/baz.h>`).

*  Avoid using UNIX directory shortcuts, such as `.` (current directory) or `..`
   (parent directory).

*  For legacy or `third_party` code that requires includes pointing outside the
   project repository, such as external repository includes requiring a prefix,
   use the [`include_prefix`](be/c-cpp.html#cc_library.include_prefix) and
   [`strip_include_prefix`](be/c-cpp.html#cc_library.strip_include_prefix)
   arguments on the `cc_library` rule target.
