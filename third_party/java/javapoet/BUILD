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

distrib_java_import(
    name = "javapoet",
    jars = ["javapoet-1.8.0.jar"],
    enable_distributions = ["debian"],
)
