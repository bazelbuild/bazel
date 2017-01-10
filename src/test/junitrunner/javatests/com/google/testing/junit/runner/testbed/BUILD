DEFAULT_VISIBILITY = [
    "//src/java_tools/junitrunner/java/com/google/testing/junit/runner:__subpackages__",
    "//src/test/junitrunner/javatests/com/google/testing/junit/runner:__subpackages__",
]

package(default_visibility = DEFAULT_VISIBILITY)

licenses(["notice"])  # Apache 2.0

java_library(
    name = "testbed",
    testonly = 1,
    srcs = glob(["*.java"]),
    deps = [
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["*.java"]) + ["BUILD"],
    visibility = ["//src:__pkg__"],
)
