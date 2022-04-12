load("//tools/distributions:distribution_rules.bzl", "distrib_java_import")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

distrib_java_import(
    name = "allocation_instrumenter",
    enable_distributions = ["debian"],
    jars = ["java-allocation-instrumenter-3.3.0.jar"],
)
