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
        "javac-9+181-r4173-1.jar",
        "javac-9+181-r4173-1.srcjar",
    ],
)

java_import(
    name = "javac",
    jars = ["javac-9+181-r4173-1.jar"],
)

filegroup(
    name = "javac_jar",
    srcs = ["javac-9+181-r4173-1.jar"],
)
