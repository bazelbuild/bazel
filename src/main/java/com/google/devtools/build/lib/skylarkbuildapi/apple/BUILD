# Description:
#   This package contains interfaces representing the Starlark "build API"
#   (but not the implementation of that API). Ultimately, this package
#   may be broken out of the Bazel package hierarchy to be standalone.
#   Thus, this package should not depend on Bazel-specific packages (only
#   those which contain pure-Starlark concepts, such as the interpreter or
#   annotation interfaces).

load("@rules_java//java:defs.bzl", "java_library")

package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "apple",
    srcs = glob(["*.java"]),
    deps = [
        "//third_party:guava",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/core",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/cpp",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/platform",
        # TODO(b/80307387): Remove dependency on Depset.
        "//src/main/java/net/starlark/java/annot",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib:syntax",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi",
        "//third_party:jsr305",
    ],
)
