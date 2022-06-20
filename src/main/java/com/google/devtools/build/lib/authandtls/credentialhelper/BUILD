load("@rules_java//java:defs.bzl", "java_library")

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "credentialhelper",
    srcs = glob(["*.java"]),
    visibility = ["//src:__subpackages__"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party:error_prone_annotations",
        "//third_party:guava",
    ],
)
