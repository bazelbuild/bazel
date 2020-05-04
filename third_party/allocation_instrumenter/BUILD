load("//tools/distributions:distribution_rules.bzl", "java_import")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

java_import(
    name = "allocation_instrumenter",
    jars = ["java-allocation-instrumenter-3.0.1.jar"],
    enable_distributions = ["debian"],
)

# alias(
#     name = "allocation_instrumenter",
#     actual = select({
#         "//src/conditions:debian_build": "@debian_java_deps//:allocation_instrumenter",
#         "//conditions:default": ":_allocation_instrumenter",
#     }),
# )

