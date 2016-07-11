package(default_visibility = ["//visibility:public"])

licenses(["restricted"])  # GNU GPL v2 with Classpath exception

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

filegroup(
    name = "test-srcs",
    srcs = select({
        "//tools/jdk:jdk7": [
            "BUILD",
            "LICENSE",
            "javac7.jar",
            "javac7.srcjar",
        ],
        "//conditions:default": [
            "BUILD",
            "LICENSE",
            "javac.jar",
            "javac.srcjar",
        ],
    }),
)

java_import(
    name = "javac",
    jars = ["javac.jar"],
)

filegroup(
    name = "javac_jar",
    srcs = select({
        "//tools/jdk:jdk7": ["javac7.jar"],
        "//conditions:default": ["javac.jar"],
    }),
)

java_import(
    name = "javac7",
    jars = ["javac7.jar"],
)

filegroup(
    name = "javac7_jar",
    srcs = ["javac7.jar"],
)
