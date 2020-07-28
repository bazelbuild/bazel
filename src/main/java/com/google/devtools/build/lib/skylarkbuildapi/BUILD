# Description:
#   This package contains interfaces representing the Starlark "build API"
#   (but not the implementation of that API). Ultimately, this package
#   may be broken out of the Bazel package hierarchy to be standalone.
#   Thus, this package should not depend on Bazel-specific packages (only
#   those which contain pure-Starlark concepts, such as the interpreter or
#   annotation interfaces).
#
# This package contains specific files for Bazel concepts, such as
# StarlarkRuleClassApi and FileApi. Only add it as a dependency if you require
# those APIs.

package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/android:srcs",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/apple:srcs",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/config:srcs",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/cpp:srcs",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/java:srcs",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/javascript:srcs",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/go:srcs",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/python:srcs",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/repository:srcs",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/stubs:srcs",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/test:srcs",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/proto:srcs",
    ],
    visibility = ["//src:__subpackages__"],
)
