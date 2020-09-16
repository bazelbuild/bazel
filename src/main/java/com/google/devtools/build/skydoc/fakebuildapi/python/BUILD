load("@rules_java//java:defs.bzl", "java_library")

package(
    default_visibility = ["//src:__subpackages__"],
)

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

java_library(
    name = "python",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/python",
        "//src/main/java/com/google/devtools/build/skydoc/fakebuildapi/config",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/java/net/starlark/java/syntax",
    ],
)
