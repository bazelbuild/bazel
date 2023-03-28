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

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/android:srcs",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/apple:srcs",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/config:srcs",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/core:srcs",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/cpp:srcs",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/go:srcs",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/java:srcs",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/objc:srcs",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/platform:srcs",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/proto:srcs",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/python:srcs",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/repository:srcs",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/stubs:srcs",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/test:srcs",
    ],
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "starlarkbuildapi",
    srcs = glob(["*.java"]),
    deps = [
        "//third_party:guava",
        "//src/main/java/net/starlark/java/annot",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/java/com/google/devtools/build/docgen/annot",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/packages/semantics",
        # TODO(cparsons): Remove dependency on Label implementation.
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/core",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/platform",
        "//third_party:jsr305",
    ],
)
