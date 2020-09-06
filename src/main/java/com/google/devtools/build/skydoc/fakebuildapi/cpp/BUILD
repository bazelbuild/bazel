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
    name = "cpp",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib:syntax",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/core",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/cpp",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/go",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/platform",
        "//src/main/java/com/google/devtools/build/skydoc/fakebuildapi",
        "//third_party:guava",
    ],
)
