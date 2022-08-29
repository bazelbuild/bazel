load("@rules_java//java:defs.bzl", "java_library")

package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "credentialhelper",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party:auth",
        "//third_party:auto_value",
        "//third_party:caffeine",
        "//third_party:error_prone_annotations",
        "//third_party:gson",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)
