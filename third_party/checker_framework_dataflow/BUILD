load("@rules_java//java:defs.bzl", "java_import")

package(default_visibility = ["//visibility:public"])

licenses(["restricted"])  # GNU GPL v2 with Classpath exception

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

java_import(
    name = "checker_framework_dataflow",
    jars = ["dataflow-shaded-3.1.2.jar"],
    srcjar = "dataflow-shaded-3.1.2-sources.jar",
    deps = [
        "//third_party/checker_framework_javacutil",
    ],
)

filegroup(
    name = "bootstrap",
    srcs = ["dataflow-shaded-3.1.2.jar"],
)
