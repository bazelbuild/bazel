load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib/server:__pkg__"],
)

java_library(
    name = "signal",
    srcs = [
        "InterruptSignalHandler.java",
    ],
    deps = ["//third_party/java/guava:base"],
)
