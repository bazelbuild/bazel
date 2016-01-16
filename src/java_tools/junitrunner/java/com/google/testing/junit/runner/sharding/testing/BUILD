DEFAULT_VISIBILITY = [
    "//java/com/google/testing/junit/runner:__subpackages__",
    "//javatests/com/google/testing/junit/runner:__subpackages__",
    "//third_party/bazel/src/java_tools/junitrunner/java/com/google/testing/junit/runner:__subpackages__",
    "//third_party/bazel/src/java_tools/junitrunner/javatests/com/google/testing/junit/runner:__subpackages__",
]

package(default_visibility = ["//src:__subpackages__"])

java_library(
    name = "testing",
    testonly = 1,
    srcs = glob(["*.java"]),
    deps = [
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/sharding",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/sharding/api",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)
