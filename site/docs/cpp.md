---
layout: documentation
title: C++ Basics
---

C++ Basics
==========

Use fully qualified include paths
---------------------------------

Includes are relative to the root of your workspace. For example, suppose
you have the following directory structure:

```
[workspace]/
    WORKSPACE
    a/
        BUILD
        a.h
        a.cc
    b/
        BUILD
        b.h
        b.cc
        main.cc
```

If `b/main.cc` needs to include b.h then we'd create the following `b/BUILD`
file:

```python
cc_library(
    name = "b",
    srcs = ["b.cc"],
    hdrs = ["b.h"],
)

cc_binary(
    name = "main",
    srcs = ["main.cc"],
    deps = [":b"],
)
```

`b/main.cc` would have the following include statement:

```cpp
#include "b/b.h"
```

Note that the full path from the package root is used. If we want `b/main.cc` to
also depend on `a/a.h`, we'd add the rule to `a/BUILD`:

```python
cc_library(
    name = "a",
    srcs = ["a.cc"],
    hdrs = ["a.h"],
    visibility = ["//b:__pkg__"],
)
```

Then we'd add a dependency to `b/BUILD`:

```python
cc_binary(
    name = "main",
    srcs = ["main.cc"],
    deps = [
        ":b",
        "//a",
    ],
)
```

And the following include to `b/main.cc`:

```cpp
#include "a/a.h"
```

`b/main.cc` will then be able to access symbols from `a/a.h` or `b/b.h`.

Transitive includes
-------------------

If a file includes a header then the file's rule should depend on that header's
library.  Conversely, only direct dependencies need to be specified as
dependencies.  For example, suppose `sandwich.h` includes `bread.h` and
`bread.h` includes `flour.h`.  `sandwich.h` doesn't include `flour.h` (who wants
flour in their sandwich?), so the BUILD file would look like:

```python
cc_library(
    name = "sandwich",
    srcs = ["sandwich.cc"],
    hdrs = ["sandwich.h"],
    deps = [":bread"],
)

cc_library(
    name = "bread",
    srcs = ["bread.cc"],
    hdrs = ["bread.h"],
    deps = [":flour"],
)

cc_library(
    name = "flour",
    srcs = ["flour.cc"],
    hdrs = ["flour.h"],
)
```

This expresses that the `sandwich` library depends on the `bread` library,
which depends on the `flour` library.

Adding include paths
--------------------

Sometimes you cannot (or do not want to) base include paths at the workspace
root. Existing libaries might already have a include directory that doesn't
match its path in your workspace.  For example, suppose you have the following
directory structure:

```
[workspace]/
    WORKSPACE
    third_party/
        some_lib/
            include/
                some_lib.h
            BUILD
            some_lib.cc
```

Bazel will expect `some_lib.h` to be included as
`third_party/some_lib/include/some_lib.h`, but suppose `some_lib.cc` includes
`"include/some_lib.h"`.  To make that include path valid,
`third_party/some_lib/BUILD` will need to specify that the `some_lib/`
directory is an include directory:

```python
cc_library(
    name = "some_lib",
    srcs = ["some_lib.cc"],
    hdrs = ["some_lib.h"],
    copts = ["-Ithird_party/some_lib"],
)
```

This is especially useful for external dependencies, as their header files
must otherwise be included with an `external/[repository-name]/` prefix.

Including external libraries: an example
----------------------------------------

Suppose you are using [Google Test](https://code.google.com/p/googletest/). You
can use one of the `new_` repository functions in the `WORKSPACE` file to
download Google Test and make it available in your repository:

```python
new_http_archive(
    name = "gtest",
    url = "https://googletest.googlecode.com/files/gtest-1.7.0.zip",
    sha256 = "247ca18dd83f53deb1328be17e4b1be31514cedfc1e3424f672bf11fd7e0d60d",
    build_file = "gtest.BUILD",
)
```

Then create `gtest.BUILD`, a BUILD file to use to compile Google Test.
Google Test has several "special" requirements that make its `cc_library` rule
more complicated:

* `gtest-1.7.0/src/gtest-all.cc` `#include`s all of the other files in
  `gtest-1.7.0/src/`, so we need to exclude it from the compile or we'll get
  link errors for duplicate symbols.
* It uses header files that relative to the `gtest-1.7.0/include/` directory
  (`"gtest/gtest.h"`), so we must add that directory the include paths.
* It needs to link in pthread, so we add that as a `linkopt`.

The final rule looks like this:

```python
cc_library(
    name = "main",
    srcs = glob(
        ["gtest-1.7.0/src/*.cc"],
        exclude = ["gtest-1.7.0/src/gtest-all.cc"]
    ),
    hdrs = glob([
        "gtest-1.7.0/include/**/*.h",
        "gtest-1.7.0/src/*.h"
    ]),
    copts = [
        "-Iexternal/gtest/gtest-1.7.0/include"
    ],
    linkopts = ["-pthread"],
    visibility = ["//visibility:public"],
)
```

This is somewhat messy: everything is prefixed with gtest-1.7.0 as a byproduct
of the archive's structure. You can make `new_http_archive` strip this prefix by
adding the `strip_prefix` attribute:

```python
new_http_archive(
    name = "gtest",
    url = "https://googletest.googlecode.com/files/gtest-1.7.0.zip",
    sha256 = "247ca18dd83f53deb1328be17e4b1be31514cedfc1e3424f672bf11fd7e0d60d",
    build_file = "gtest.BUILD",
    strip_prefix = "gtest-1.7.0",
)
```

Then `gtest.BUILD` would look like this:

```python
cc_library(
    name = "main",
    srcs = glob(
        ["src/*.cc"],
        exclude = ["src/gtest-all.cc"]
    ),
    hdrs = glob([
        "include/**/*.h",
        "src/*.h"
    ]),
    copts = ["-Iexternal/gtest/include"],
    linkopts = ["-pthread"],
    visibility = ["//visibility:public"],
)
```

Now `cc_` rules can depend on `//external:gtest/main`.

For example, we could create a test such as:

```cpp
#include "gtest/gtest.h"

TEST(FactorialTest, Negative) {
  EXPECT_EQ(1, 1);
}
```

Then create a BUILD file for your tests:

```python
cc_test(
    name = "my_test",
    srcs = ["my_test.cc"],
    copts = ["-Iexternal/gtest/include"],
    deps = ["@gtest//:main"],
)
```

You can then use `bazel test` to run the test.


Adding dependencies on precompiled libraries
--------------------------------------------

If you want to use a library that you only have a compiled version of (e.g.,
headers and a .so) wrap it in a `cc_library` rule:

```python
cc_library(
    name = "mylib",
    srcs = ["mylib.so"],
    hdrs = ["mylib.h"],
)
```

Then other C++ targets in your workspace can depend on this rule.
