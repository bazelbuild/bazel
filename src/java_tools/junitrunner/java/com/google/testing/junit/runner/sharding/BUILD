package(default_visibility = ["//src:__subpackages__"])

# TODO(bazel-team): This should be testonly = 1.
java_library(
    name = "sharding",
    srcs = glob(["*.java"]),
    deps = [
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/sharding/api",
        "//src/java_tools/junitrunner/java/com/google/testing/junit/runner/util",
        "//third_party:jsr305",
        "//third_party:jsr330_inject",
        "//third_party:junit4",
    ],
)

filegroup(
    name = "srcs",
    testonly = 0,  # All srcs should be not test only, overwrite package default.
    srcs = glob(["**"]),
)
