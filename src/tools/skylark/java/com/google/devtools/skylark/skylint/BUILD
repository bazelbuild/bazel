# The Skylint linter for BUILD and Skylark files
# Open-sourced as part of Bazel.

java_binary(
    name = "Skylint",
    srcs = glob(["**/*.java"]),
    main_class = "com.google.devtools.skylark.skylint.Skylint",
    deps = [
        ":skylint-lib",
        "//src/main/java/com/google/devtools/build/lib:packages",
    ],
)

java_library(
    name = "skylint-lib",
    srcs = glob(["**/*.java"]),
    visibility = ["//src/tools/skylark/javatests/com/google/devtools/skylark/skylint:__pkg__"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:packages",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(
        ["**"],
    ),
    visibility = ["//src:__pkg__"],
)
