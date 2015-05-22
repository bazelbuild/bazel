---
layout: documentation
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

If _b/main.cc_ needs to include b.h then we'd create the following _b/BUILD_
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

_b/main.cc_ would have the following include statement:

```cpp
#include "b/b.h"
```

Note that the full path from the package root is used. If we want _b/main.cc_ to
also depend on _a/a.h_, we'd add the rule to _a/BUILD_:

```python
cc_library(
    name = "a",
    srcs = ["a.cc"],
    hdrs = ["a.h"],
    visibility = ["//b:__pkg__"],
)
```

Then we'd add a dependency to _b/BUILD_:

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

And the following include to _b/main.cc_:

```cpp
#include "a/a.h"
```

_b/main.cc_ will then be able to access symbols from _a/a.h_ or _b/b.h_.

Transitive includes
-------------------

If a file includes a header then the file's rule should depend on that header's
library.  Conversely, only direct dependencies need to be specified as
dependencies.  For example, suppose _sandwich.h_ includes _bread.h_ and
_bread.h_ includes _flour.h_.  _sandwich.h_ doesn't include _flour.h_ (who wants
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

This expresses that the "sandwich" library depends on the "bread" library,
which depends on the "flour" library.

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

Bazel will expect _some_lib.h_ to be included as
`third_party/some_lib/include/some_lib.h`, but suppose _some_lib.cc_ includes
`"include/some_lib.h"`.  To make that include path valid,
_third\_party/some_lib/BUILD_ will need to specify that the _some_lib/_
directory is an include directory:

```python
cc_library(
    name = "some_lib",
    srcs = ["some_lib.cc"],
    hdrs = ["some_lib.h"],
    includes = ["."],
)
```

This is especially useful for external dependencies, as their header files
must otherwise be included with an "external/[repository-name]/" prefix.

Including external libraries: an example
----------------------------------------

Suppose you are using [Google Test](https://code.google.com/p/googletest/). You
can use one of the `new_` repository functions in the _WORKSPACE_ file to
download Google Test and make it available in your repository:

```python
new_http_archive(
    name = "gtest",
    url = "https://googletest.googlecode.com/files/gtest-1.7.0.zip",
    sha256 = "247ca18dd83f53deb1328be17e4b1be31514cedfc1e3424f672bf11fd7e0d60d",
    build_file = "gtest.BUILD",
)
```

Then create _gtest.BUILD_, a BUILD file to use to compile Google Test.
Google Test has several "special" requirements that make its `cc_library` rule
more complicated:

* _gtest-1.7.0/src/gtest-all.cc_ `#include`s all of the other files in
  _gtest-1.7.0/src/_, so we need to exclude it from the compile or we'll get
  link errors for duplicate symbols.
* It uses header files that relative to the _gtest-1.7.0/include/_ directory
  (`"gtest/gtest.h"`), so we must add that directory the includes.
* It uses "private" header files in src/, so we add "." to the includes so it
  can `#include "src/gtest-internal-inl.h"`.
* It needs to link in pthread, so we add that as a `linkopt`.

The final rule looks like this:

```python
cc_library(
    name = "main",
    srcs = glob(
        ["gtest-1.7.0/src/*.cc"],
        exclude = ["gtest-1.7.0/src/gtest-all.cc"]
    ),
    hdrs = glob(["gtest-1.7.0/include/**/*.h"]),
    includes = [
        "gtest-1.7.0",
        "gtest-1.7.0/include"
    ],
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
    deps = ["@gtest//:main"],
)
```

You can then use `bazel test` to run the test.
