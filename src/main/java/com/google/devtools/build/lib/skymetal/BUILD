load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src/main/java/com/google/devtools/build/lib:build_lib"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib:__pkg__"],
)

java_library(
    name = "skymetal_bridge",
    srcs = ["SkymetalBridge.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/bugreport",
        "//src/main/java/com/google/devtools/build/lib/jni",
        "//third_party/java/error_prone:annotations",
        "//third_party/java/guava:base",
    ],
)

java_library(
    name = "skymetal_feature",
    srcs = ["SkymetalFeature.java"],
    deps = ["//third_party/java/flogger"],
)
