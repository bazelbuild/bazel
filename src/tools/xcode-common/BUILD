package(default_visibility = ["//src/main/java:__subpackages__"])

# TODO(bazel-team): Split this into multiple rules.
java_library(
    name = "xcode-common",
    srcs = glob([
        "java/com/google/devtools/build/xcode/util/*.java",
        "java/com/google/devtools/build/xcode/common/*.java",
    ]),
    deps = [
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/test:__subpackages__"],
)
