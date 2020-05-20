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

load("@rules_java//java:defs.bzl", "java_library")

package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/android:srcs",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/apple:srcs",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/config:srcs",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/core:srcs",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/cpp:srcs",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/java:srcs",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/javascript:srcs",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/go:srcs",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/platform:srcs",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/python:srcs",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/repository:srcs",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/stubs:srcs",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/test:srcs",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/proto:srcs",
    ],
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "skylarkbuildapi",
    srcs = glob(["*.java"]),
    deps = [
        "//third_party:guava",
        "//src/main/java/net/starlark/java/annot",
        "//src/main/java/com/google/devtools/build/lib:syntax",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        # TODO(cparsons): Remove dependency on Label implementation.
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/core",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/platform",
        "//third_party:jsr305",
    ],
)
