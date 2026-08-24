load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "http",
    srcs = glob(["*.java"]),
    visibility = [
        "//src/test/java/com/google/devtools/build/lib/remote/http:__pkg__",
        "//src/tools/remote:__subpackages__",
    ],
    deps = [
        "//third_party/java/flogger",
        "//third_party/java/guava:annotations",
        "//third_party/java/guava:base",
        "//third_party/java/jsr305_annotations",
        "//third_party/java/netty4:buffer",
        "//third_party/java/netty4:codec_http",
        "//third_party/java/netty4:common",
        "//third_party/java/netty4:transport",
    ],
)
