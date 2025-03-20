load("@rules_java//java:defs.bzl", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_test(
    name = "SpellCheckerTest",
    srcs = ["SpellCheckerTest.java"],
    deps = [
        "//src/main/java/net/starlark/java/spelling",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)
