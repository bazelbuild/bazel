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
            "javac-9-dev-r3297-4.jar",
            "javac-9-dev-r3297-4.srcjar",
        ],
    }),
)

java_import(
    name = "javac",
    jars = ["javac-9-dev-r3297-4.jar"],
)

filegroup(
    name = "javac_jar",
    srcs = select({
        "//tools/jdk:jdk7": ["javac7.jar"],
        "//conditions:default": ["javac-9-dev-r3297-4.jar"],
    }),
)

filegroup(
    name = "test-srcs-9-dev-r4023-1",
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
            "javac-9-dev-r4023-1.jar",
            "javac-9-dev-r4023-1.srcjar",
        ],
    }),
)

java_import(
    name = "javac-9-dev-r4023-1",
    jars = ["javac-9-dev-r4023-1.jar"],
)

filegroup(
    name = "javac_jar-9-dev-r4023-1",
    srcs = select({
        "//tools/jdk:jdk7": ["javac7.jar"],
        "//conditions:default": ["javac-9-dev-r4023-1.jar"],
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
