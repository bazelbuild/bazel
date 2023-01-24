load("//tools/distributions:distribution_rules.bzl", "distrib_java_import")

licenses(["notice"])  # Apache License 2.0

exports_files(["LICENSE"])

package(
    default_visibility = ["//visibility:public"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

java_library(
    name = "javapoet",
    exports = ["@maven//:com_squareup_javapoet"],
)
