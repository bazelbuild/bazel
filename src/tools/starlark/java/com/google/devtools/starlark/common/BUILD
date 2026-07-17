load("@rules_java//java:defs.bzl", "java_library")

java_library(
    name = "common",
    srcs = ["DocstringUtils.java"],
    visibility = [
        # For docstring parsing libraries.
        "//src/main/java/com/google/devtools/build/lib/starlarkdocextract:__subpackages__",
        "//src/tools/skylark/java/com/google/devtools/skylark/skylint:__subpackages__",
        "//src/tools/skylark/javatests/com/google/devtools/skylark/skylint:__subpackages__",
    ],
    deps = [
        "//src/main/java/net/starlark/java/annot",
        "//third_party:guava",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(
        ["**"],
    ),
    visibility = ["//src:__pkg__"],
)
