package(default_visibility = ["//src:__subpackages__"])

# TODO(bazel-team): This should be testonly = 1.
java_library(
    name = "api",
    srcs = glob(["*.java"]),
    deps = [
        "//third_party:junit4",
    ],
)

filegroup(
    name = "srcs",
    testonly = 0,  # All srcs should be not test only, overwrite package default.
    srcs = glob(["**"]),
)
