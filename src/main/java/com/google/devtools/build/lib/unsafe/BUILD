package(
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

java_library(
    name = "unsafe-provider",
    srcs = ["UnsafeProvider.java"],
)

java_library(
    name = "string",
    srcs = [
        "RuntimeVersion.java",
        "StringUnsafe.java",
    ],
    deps = [
        ":unsafe-provider",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)
