---
layout: documentation
title: Build C++
---

Build C++
=========

You can use Bazel to build your C++ application. In this tutorial you'll learn how to:

* Build your first C++ target
* Use external libraries
* Write and run C++ tests
* Use precompiled libraries

## Setting up your workspace

Suppose that you have an existing project in a directory, say,
`~/gitroot/my-project/`. Create an empty file at
`~/gitroot/my-project/WORKSPACE` to show Bazel where your project's root is.
We are going to create a small hello world project with the following directory structure:
{% highlight bash %}
└── my-project
    ├── lib
    │   ├── BUILD
    │   ├── hello-greet.cc
    │   └── hello-greet.h
    ├── main
    │   ├── BUILD
    │   ├── hello-time.cc
    │   ├── hello-time.h
    │   └── hello-world.cc
    └── WORKSPACE
{% endhighlight %}

## Creating source files

Using the following commands to create the necessary source files:
{% highlight bash %}
# If you're not already there, move to your workspace directory.
cd ~/gitroot/my-project
mkdir ./main
cat > main/hello-world.cc <<'EOF'

#include "lib/hello-greet.h"
#include "main/hello-time.h"
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  std::string who = "world";
  if (argc > 1) {
    who = argv[1];
  }
  std::cout << get_greet(who) <<std::endl;
  print_localtime();
  return 0;
}
EOF

cat > main/hello-time.h <<'EOF'

#ifndef MAIN_HELLO_TIME_H_
#define MAIN_HELLO_TIME_H_

void print_localtime();

#endif
EOF

cat > main/hello-time.cc <<'EOF'

#include "main/hello-time.h"
#include <ctime>
#include <iostream>

void print_localtime() {
  std::time_t result = std::time(nullptr);
  std::cout << std::asctime(std::localtime(&result));
}
EOF

mkdir ./lib
cat > lib/hello-greet.h <<'EOF'

#ifndef LIB_HELLO_GREET_H_
#define LIB_HELLO_GREET_H_

#include <string>

std::string get_greet(const std::string &thing);

#endif
EOF

cat > lib/hello-greet.cc <<'EOF'

#include "lib/hello-greet.h"
#include <string>

std::string get_greet(const std::string& who) {
  return "Hello " + who;
}
EOF
{% endhighlight %}

## Adding BUILD files

As you can see from the source code, `main/hello-world.cc` needs to include both `lib/hello-greet.h` and `main/hello-time.h`.
First we create `lib/BUILD` for hello-greet.cc:

{% highlight python %}
cc_library(
    name = "hello-greet",
    srcs = ["hello-greet.cc"],
    hdrs = ["hello-greet.h"],
    visibility = ["//main:__pkg__"],
)
{% endhighlight %}

Note that `visibility = ["//main:__pkg__"]` indicates `hello-greet` is visible from `main/BUILD`.
Then we'd create the following `main/BUILD` file:

{% highlight python %}
cc_library(
    name = "hello-time",
    srcs = ["hello-time.cc"],
    hdrs = ["hello-time.h"],
)

cc_binary(
    name = "hello-world",
    srcs = ["hello-world.cc"],
    deps = [
        ":hello-time",
        "//lib:hello-greet",
    ],
)
{% endhighlight %}

Note when depending on a target in the same package, we can just use `:hello-time`.
When the target is in other package, a full path from root should be used, like `//lib:hello-greet`.

Now you are ready to build your hello world C++ binary:

{% highlight bash %}
bazel build main:hello-world
{% endhighlight %}

This produces the following output:

{% highlight bash %}
INFO: Found 1 target...
Target //main:hello-world up-to-date:
  bazel-bin/main/hello-world
INFO: Elapsed time: 2.869s, Critical Path: 1.00s
{% endhighlight %}

{% highlight bash %}
./bazel-bin/main/hello-world
{% endhighlight %}

This produces the following output:

{% highlight bash %}
Hello world
Thu Jun 23 18:51:46 2016
{% endhighlight %}

{% highlight bash %}
./bazel-bin/main/hello-world Bazel
{% endhighlight %}

This produces the following output:

{% highlight bash %}
Hello Bazel
Thu Jun 23 18:52:10 2016
{% endhighlight %}

Congratulations, you've just built your first Bazel target!

## Transitive includes

If a file includes a header, then the file's rule should depend on that header's
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

Here, the `sandwich` library depends on the `bread` library, which depends
on the `flour` library.

## Adding include paths

Sometimes you cannot (or do not want to) base include paths at the workspace
root. Existing libraries might already have a include directory that doesn't
match its path in your workspace.  For example, suppose you have the following
directory structure:

```
└── my-project
    ├── third_party
    │   └── some_lib
    │       ├── BUILD
    │       ├── include
    │       │   └── some_lib.h
    │       └── some_lib.cc
    └── WORKSPACE
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

## Including external libraries

Suppose you are using [Google Test](https://github.com/google/googletest). You
can use one of the `new_` repository functions in the `WORKSPACE` file to
download Google Test and make it available in your repository:

```python
new_http_archive(
    name = "gtest",
    url = "https://github.com/google/googletest/archive/release-1.7.0.zip",
    sha256 = "b58cb7547a28b2c718d1e38aee18a3659c9e3ff52440297e965f5edffe34b6d0",
    build_file = "gtest.BUILD",
)
```

Then create `gtest.BUILD`, a BUILD file to use to compile Google Test.
Google Test has several "special" requirements that make its `cc_library` rule
more complicated:

* `googletest-release-1.7.0/src/gtest-all.cc` `#include`s all of the other files in
  `googletest-release-1.7.0/src/`, so we need to exclude it from the compile or we'll get
  link errors for duplicate symbols.
* It uses header files that are relative to the `googletest-release-1.7.0/include/` directory
  (`"gtest/gtest.h"`), so we must add that directory to the include paths.
* It needs to link in pthread, so we add that as a `linkopt`.

The final rule looks like this:

```python
cc_library(
    name = "main",
    srcs = glob(
        ["googletest-release-1.7.0/src/*.cc"],
        exclude = ["googletest-release-1.7.0/src/gtest-all.cc"]
    ),
    hdrs = glob([
        "googletest-release-1.7.0/include/**/*.h",
        "googletest-release-1.7.0/src/*.h"
    ]),
    copts = [
        "-Iexternal/gtest/googletest-release-1.7.0/include"
    ],
    linkopts = ["-pthread"],
    visibility = ["//visibility:public"],
)
```

This is somewhat messy: everything is prefixed with googletest-release-1.7.0 as a byproduct
of the archive's structure. You can make `new_http_archive` strip this prefix by
adding the `strip_prefix` attribute:

```python
new_http_archive(
    name = "gtest",
    url = "https://github.com/google/googletest/archive/release-1.7.0.zip",
    sha256 = "b58cb7547a28b2c718d1e38aee18a3659c9e3ff52440297e965f5edffe34b6d0",
    build_file = "gtest.BUILD",
    strip_prefix = "googletest-release-1.7.0",
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

Now `cc_` rules can depend on `@gtest//:main`.

## Writing and running C++ tests

For example, we could create a test `./test/hello-test.cc` such as:

```cpp
#include "gtest/gtest.h"
#include "lib/hello-greet.h"

TEST(HelloTest, GetGreet) {
  EXPECT_EQ(get_greet("Bazel"), "Hello Bazel");
}
```

Then create `./test/BUILD` file for your tests:

```python
cc_test(
    name = "hello-test",
    srcs = ["hello-test.cc"],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        "@gtest//:main",
        "//lib:hello-greet",
    ],
)
```

Note in order to make `hello-greet` visible to `hello-test`, we have to add `"//test:__pkg__",` to `visibility` attribute in `./lib/BUILD`.

Now you can use `bazel test` to run the test.

{% highlight bash %}
bazel test test:hello-test
{% endhighlight %}

This produces the following output:

{% highlight bash %}
INFO: Found 1 test target...
Target //test:hello-test up-to-date:
  bazel-bin/test/hello-test
INFO: Elapsed time: 4.497s, Critical Path: 2.53s
//test:hello-test                                                        PASSED in 0.3s

Executed 1 out of 1 tests: 1 test passes.
{% endhighlight %}


## Adding dependencies on precompiled libraries

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
