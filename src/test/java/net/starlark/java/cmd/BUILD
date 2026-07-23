load("@rules_java//java:defs.bzl", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_test(
    name = "MainTest",
    srcs = ["MainTest.java"],
    deps = [
        "//src/main/java/net/starlark/java/cmd:cmd_lib",
        "//src/main/java/net/starlark/java/eval",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)
