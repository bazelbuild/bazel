load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
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
    srcs = ["StringUnsafe.java"],
    deps = [":unsafe-provider"],
)
