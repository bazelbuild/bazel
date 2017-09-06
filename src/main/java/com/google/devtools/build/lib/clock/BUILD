package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib:__pkg__"],
)

# Clock classes.
java_library(
    name = "clock",
    srcs = glob([
        "*.java",
    ]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib/concurrent",
    ],
)
