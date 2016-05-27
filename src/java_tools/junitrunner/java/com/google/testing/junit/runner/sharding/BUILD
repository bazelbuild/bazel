DEFAULT_VISIBILITY = [
    "//java/com/google/testing/junit/runner:__subpackages__",
    "//javatests/com/google/testing/junit/runner:__subpackages__",
    "//third_party/bazel/src/java_tools/junitrunner/java/com/google/testing/junit/runner:__subpackages__",
    "//third_party/bazel/src/java_tools/junitrunner/javatests/com/google/testing/junit/runner:__subpackages__",
]

package(default_visibility = ["//src:__subpackages__"])

# TODO(bazel-team): This should be testonly = 1.
java_library(
    name = "sharding",
    srcs = glob(["*.java"]),
    deps = [
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/sharding/api",
        "//third_party:guava",
        "//third_party:guice",
        "//third_party:jsr305",
        "//third_party:junit4",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)
