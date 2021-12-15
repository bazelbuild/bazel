load("@rules_java//java:defs.bzl", "java_import")

package(default_visibility = ["//visibility:public"])

licenses(["restricted"])  # GNU GPL v2 with Classpath exception

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

java_import(
    name = "checker_framework_dataflow",
    jars = ["dataflow-errorprone-3.15.0.jar"],
    srcjar = "dataflow-errorprone-3.15.0-sources.jar",
)

filegroup(
    name = "bootstrap",
    srcs = ["dataflow-errorprone-3.15.0.jar"],
)
