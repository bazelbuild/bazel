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

# Library of collection utilities.
java_library(
    name = "compacthashset",
    srcs = ["CompactHashSet.java"],
    deps = [
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)
