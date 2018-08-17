package(
    default_testonly = 1,
    default_visibility = [
        "//javatests/com/google/devtools/build/lib:__subpackages__",
        "//src/test/java/com/google/devtools/build/lib:__subpackages__",
    ],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]) + [
        "//src/test/java/com/google/devtools/build/lib/blackbox/framework:srcs",
        "//src/test/java/com/google/devtools/build/lib/blackbox/junit:srcs",
    ],
    visibility = ["//src/test/java/com/google/devtools/build/lib:__pkg__"],
)
