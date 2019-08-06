load("@rules_java//java:defs.bzl", "java_library")

package(default_visibility = ["//visibility:public"])

licenses(["restricted"])  # GNU GPL v2 with Classpath exception

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

java_library(
    name = "jformatstring",
    srcs = glob(["java/**"]),
)

load("//tools/build_rules:java_rules_skylark.bzl", "bootstrap_java_library")

bootstrap_java_library(
    name = "bootstrap",
    srcs = glob(["java/**"]),
)
