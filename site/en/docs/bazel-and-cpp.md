Project: /_project.yaml
Book: /_book.yaml

# C++ and Bazel

{% include "_buttons.html" %}

This page contains resources that help you use Bazel with C++ projects. It links
to a tutorial, build rules, and other information specific to building C++
projects with Bazel.

## Working with Bazel {:#working-with-bazel}

The following resources will help you work with Bazel on C++ projects:

*  [Tutorial: Building a C++ project](/start/cpp)
*  [C++ common use cases](/tutorials/cpp-use-cases)
*  [C/C++ rules](/reference/be/c-cpp)
*  Essential Libraries
   -  [Abseil](https://abseil.io/docs/cpp/quickstart){: .external}
   -  [Boost](https://github.com/nelhage/rules_boost){: .external}
   -  [HTTPS Requests: CPR and libcurl](https://github.com/hedronvision/bazel-make-cc-https-easy){: .external}
*  [C++ toolchain configuration](/docs/cc-toolchain-config-reference)
*  [Tutorial: Configuring C++ toolchains](/tutorials/ccp-toolchain-config)
*  [Integrating with C++ rules](/configure/integrate-cpp)

## Best practices {:#best-practices}

In addition to [general Bazel best practices](/configure/best-practices), below are
best practices specific to C++ projects.

### BUILD files {:#build-files}

Follow the guidelines below when creating your BUILD files:

*  Each `BUILD` file should contain one [`cc_library`](/reference/be/c-cpp#cc_library)
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

### Include paths {:#include-paths}

Follow these guidelines for include paths:

*  Make all include paths relative to the workspace directory.

*  Use quoted includes (`#include "foo/bar/baz.h"`) for non-system headers, not
   angle-brackets (`#include <foo/bar/baz.h>`).

*  Avoid using UNIX directory shortcuts, such as `.` (current directory) or `..`
   (parent directory).

*  For legacy or `third_party` code that requires includes pointing outside the
   project repository, such as external repository includes requiring a prefix,
   use the [`include_prefix`](/reference/be/c-cpp#cc_library.include_prefix) and
   [`strip_include_prefix`](/reference/be/c-cpp#cc_library.strip_include_prefix)
   arguments on the `cc_library` rule target.

### Toolchain features {:#toolchain-features}

The following optional [features](/docs/cc-toolchain-config-reference#features)
can improve the hygiene of a C++ project. They can be enabled using the
`--features` command-line flag or the `features` attribute of
[`repo`](/external/overview#repo.bazel),
[`package`](/reference/be/functions#package) or `cc_*` rules:

* The `parse_headers` feature makes it so that the C++ compiler is used to parse
  (but not compile) all header files in the built targets and their dependencies
  when using the
  [`--process_headers_in_dependencies`](/reference/command-line-reference#flag--process_headers_in_dependencies)
  flag. This can help catch issues in header-only libraries and ensure that
  headers are self-contained and independent of the order in which they are
  included. When the `parse_headers_as_c` feature is also enabled, headers are
  parsed as C rather than C++.
* The `layering_check` feature enforces that targets only include headers
  provided by their direct dependencies. The default toolchain supports this
  feature on Linux with `clang` as the compiler.
