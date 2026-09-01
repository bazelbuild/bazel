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

java_library(
    name = "supplier",
    srcs = glob(["*.java"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib:build_lib"],
    deps = [
        "//third_party/java/error_prone:annotations",
        "//third_party/java/guava:annotations",
        "//third_party/java/guava:base",
        "//third_party/java/jsr305_annotations",
    ],
)
