package(default_visibility = ["//visibility:public"])

licenses(["restricted"])  # GNU GPL v2 with Classpath exception

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

filegroup(
    name = "test-srcs",
    srcs = [
        "BUILD",
        "LICENSE",
        "javac-9-dev-r4023-3.jar",
        "javac-9-dev-r4023-3.srcjar",
    ],
)

java_import(
    name = "javac",
    jars = ["javac-9-dev-r4023-3.jar"],
)

filegroup(
    name = "javac_jar",
    srcs = ["javac-9-dev-r4023-3.jar"],
)
