load("@rules_java//java:defs.bzl", "java_library")

package(default_visibility = ["//src:__subpackages__"])

licenses(["notice"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "credentialhelper",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party:auto_value",
        "//third_party:error_prone_annotations",
        "//third_party:gson",
        "//third_party:guava",
    ],
)
