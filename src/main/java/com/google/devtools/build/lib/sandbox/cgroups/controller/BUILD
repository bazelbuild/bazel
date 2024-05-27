load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob([
        "**",
        "v1/**",
        "v2/**",
    ]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "controller",
    srcs = [
        "Controller.java",
    ],
)

java_library(
    name = "v2",
    srcs = glob(["v2/**"]),
    deps = [
        ":controller",
        "//third_party:guava",
    ],
)

java_library(
    name = "v1",
    srcs = glob(["v1/**"]),
    deps = [
        ":controller",
    ],
)
