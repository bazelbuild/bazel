package(default_visibility = ["//visibility:public"])

licenses(["restricted"])  # GNU GPL v2 with Classpath exception

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

java_library(
    name = "checker_framework_javacutil",
    srcs = glob(["java/**"]),
    deps = ["@local_jdk//:langtools-neverlink"],
)

load("//tools/build_rules:java_rules_skylark.bzl", "bootstrap_java_library")

bootstrap_java_library(
    name = "bootstrap",
    srcs = glob(["java/**"]),
    neverlink_jars = ["@local_jdk//:langtools"],
)
