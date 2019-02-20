java_library(
    name = "common",
    srcs = glob(["**/*.java"]),
    visibility = [
        # For docstring parsing libraries.
        "//src/main/java/com/google/devtools/build/skydoc:__subpackages__",
        "//src/tools/skylark/java/com/google/devtools/skylark/skylint:__subpackages__",
        "//src/tools/skylark/javatests/com/google/devtools/skylark/skylint:__subpackages__",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:skylarkinterface",
        "//src/main/java/com/google/devtools/build/lib:syntax",
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
