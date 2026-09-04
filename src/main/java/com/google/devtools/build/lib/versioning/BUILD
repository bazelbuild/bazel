load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib:__pkg__"],
)

java_library(
    name = "long_version_getter",
    srcs = ["LongVersionGetter.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/vfs",
    ],
)
