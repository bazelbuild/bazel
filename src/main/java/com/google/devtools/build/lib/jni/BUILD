load("@rules_java//java:defs.bzl", "java_library")

package(
    default_visibility = ["//src:__subpackages__"],
)

licenses(["notice"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "jni",
    srcs = ["JniLoader.java"],
    data = [
        "//src/main/native:target-os-unix-native-lib",
        "//src/main/native/windows:target-os-native-lib",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//third_party:flogger",
        "@bazel_tools//tools/java/runfiles",
    ],
)
