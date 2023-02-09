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
    exported_plugins = [":auto_bazel_repository_processor"],
    visibility = ["//tools/java/runfiles/testing:__pkg__"],
    exports = [":auto_bazel_repository"],
)

java_library(
    name = "auto_bazel_repository",
    srcs = ["AutoBazelRepository.java"],
)

java_plugin(
    name = "auto_bazel_repository_processor",
    srcs = ["AutoBazelRepositoryProcessor.java"],
    processor_class = "com.google.devtools.build.runfiles.AutoBazelRepositoryProcessor",
)
