load("@rules_java//java:defs.bzl", "java_library")

package(
    default_visibility = ["//src:__subpackages__"],
)

licenses(["notice"])

filegroup(
    name = "srcs",
    srcs = glob(["*"]),
    visibility = ["//src:__subpackages__"],
)

# This package defines the StarlarkSemantics used by Bazel, and its command-line flag parser.
java_library(
    name = "semantics",
    srcs = ["BuildLanguageOptions.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/syntax:evaluator",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
    ],
)
