DEFAULT_VISIBILITY = [
    "//java/com/google/testing/junit/runner:__subpackages__",
    "//javatests/com/google/testing/junit/runner:__subpackages__",
    "//third_party/bazel/src/java_tools/junitrunner/java/com/google/testing/junit/runner:__subpackages__",
    "//third_party/bazel/src/java_tools/junitrunner/javatests/com/google/testing/junit/runner:__subpackages__",
]

package(default_visibility = ["//src:__subpackages__"])

java_library(
    name = "util",
    srcs = glob(["*.java"]),
    deps = [
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:junit4",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)
