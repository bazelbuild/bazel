package(default_visibility = ["//visibility:public"])

licenses(["restricted"])  # GNU GPL v2 with Classpath exception

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

java_import(
    name = "checker_framework_javacutil",
    jars = ["javacutil-2.5.3.jar"],
    srcjar = "javacutil-2.5.3-sources.jar",
)

filegroup(
    name = "bootstrap",
    srcs = ["javacutil-2.5.3.jar"],
)
