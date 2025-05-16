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
    name = "compacthashmap",
    srcs = ["CompactHashMap.java"],
    deps = [
        "//third_party:error_prone_annotations",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)
