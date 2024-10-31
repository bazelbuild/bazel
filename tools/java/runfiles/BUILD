load("@rules_java//java:defs.bzl", "java_library")

package(default_visibility = ["//visibility:private"])

filegroup(
    name = "srcs",
    srcs = glob(
        ["**"],
        exclude = [
            ".*",
            "*~",
        ],  # .swp files and such
    ) + ["//tools/java/runfiles/testing:srcs"],
    visibility = ["//tools/java:__pkg__"],
)

filegroup(
    name = "embedded_tools",
    srcs = [
        "BUILD.tools",
        ":java-srcs",
    ],
    visibility = ["//tools/java:__pkg__"],
)

filegroup(
    name = "java-srcs",
    srcs = [
        "AutoBazelRepository.java",
        "AutoBazelRepositoryProcessor.java",
        "Runfiles.java",
        "Util.java",
    ],
)

java_library(
    name = "runfiles",
    srcs = [
        "Runfiles.java",
        "Util.java",
    ],
    visibility = [
        "//src/test/java/com/google/devtools/build/lib/bazel/rules/sh:__pkg__",
        "//src/tools/remote/src/main/java/com/google/devtools/build/remote/worker:__pkg__",
        "//tools/java/runfiles/testing:__pkg__",
    ],
)
