# Description:
#   This package contains interfaces representing the skylark "build API"
#   (but not the implementation of that API). Ultimately, this package
#   may be broken out of the Bazel package hierarchy to be standalone.
#   Thus, this package should not depend on Bazel-specific packages (only
#   those which contain pure-Skylark concepts, such as the interpreter or
#   annotation interfaces).
#
# This package contains core types that will be needed by other APIs, such as
# StructApi and ProviderApi. Only add new types that will be widely used by
# other parts of the skylarkbuildapi.

load("@rules_java//java:defs.bzl", "java_library")

package(default_visibility = ["//src:__subpackages__"])

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

java_library(
    name = "core",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:syntax",
        "//src/main/java/com/google/devtools/build/lib/skylarkinterface",
        "//third_party:guava",
    ],
)
