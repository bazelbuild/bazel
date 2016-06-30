package(default_visibility = ["//visibility:public"])

licenses(["restricted"])  # GNU GPL v2 with Classpath exception

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

java_library(
    name = "checker_framework_dataflow",
    srcs = glob(["java/**"]),
    deps = [
        "//third_party/checker_framework_javacutil",
        "@local_jdk//:langtools-neverlink",
    ],
)

load("//tools/build_rules:java_rules_skylark.bzl", "bootstrap_java_library")

bootstrap_java_library(
    name = "bootstrap",
    srcs = glob(["java/**"]),
    deps = ["//third_party/checker_framework_javacutil:bootstrap"],
)
