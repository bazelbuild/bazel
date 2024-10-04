load("@rules_java//java:defs.bzl", "java_library", "java_plugin")

java_library(
    name = "runfiles",
    srcs = [
        "Runfiles.java",
        "Util.java",
    ],
    exported_plugins = [":auto_bazel_repository_processor"],
    visibility = ["//visibility:public"],
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
