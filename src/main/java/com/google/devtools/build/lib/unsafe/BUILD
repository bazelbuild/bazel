load("@rules_java//java:defs.bzl", "java_library")

package(
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
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
