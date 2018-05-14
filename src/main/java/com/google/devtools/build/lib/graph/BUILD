package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib:__pkg__"],
)

# This package provides a general-purpose directed graph utility class with
# implementations of numerous textbook graph algorithms and code to read and
# write AT&T GraphViz files.
java_library(
    name = "graph",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib/collect/compacthashset",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)
