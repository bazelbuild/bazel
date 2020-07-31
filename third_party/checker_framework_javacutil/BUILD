load("@rules_java//java:defs.bzl", "java_import")

package(default_visibility = ["//visibility:public"])

licenses(["restricted"])  # GNU GPL v2 with Classpath exception

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

java_import(
    name = "checker_framework_javacutil",
    jars = ["javacutil-3.2.0.jar"],
    srcjar = "javacutil-3.2.0-sources.jar",
)

filegroup(
    name = "bootstrap",
    srcs = ["javacutil-3.2.0.jar"],
)
