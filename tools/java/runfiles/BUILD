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
    name = "bzl_srcs",
    srcs = glob(["*.bzl"]),
    visibility = ["//tools:__pkg__"],
)

filegroup(
    name = "embedded_tools",
    srcs = [
        "BUILD.tools",
        "RunfilesConstants.java",
        "RunfilesConstants.java.tpl",
        "java_current_repository.bzl",
        ":java-srcs",
    ],
    visibility = ["//tools/java:__pkg__"],
)

filegroup(
    name = "java-srcs",
    srcs = [
        "Runfiles.java",
        "Util.java",
    ],
)

java_library(
    name = "runfiles",
    srcs = [":java-srcs"],
    visibility = ["//tools/java/runfiles/testing:__pkg__"],
)
