package(default_visibility = ["//visibility:public"])

licenses(["restricted"])  # GNU GPL v2 with Classpath exception

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

java_import(
    name = "checker_framework_dataflow",
    jars = ["dataflow-2.5.3.jar"],
    srcjar = "dataflow-2.5.3-sources.jar",
    deps = [
        "//third_party/checker_framework_javacutil",
    ],
)

filegroup(
    name = "bootstrap",
    srcs = ["dataflow-2.5.3.jar"],
)
