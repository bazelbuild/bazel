package(default_visibility = ["//visibility:public"])

licenses(["restricted"])  # GNU GPL v2 with Classpath exception

filegroup(
    name = "srcs",
    srcs = ["javacutil-2.2.2-sources.jar"],
)

java_import(
    name = "checker_framework_javacutil",
    jars = [
        "javacutil-2.2.2.jar",
    ],
    srcjar = "javacutil-2.2.2-sources.jar",
)
