package(default_visibility = ["//visibility:public"])

licenses(["restricted"])  # GNU GPL v2 with Classpath exception

filegroup(
    name = "srcs",
    srcs = select({
        "//tools/jdk:jdk7": [
            "BUILD",
            "javac7.jar",
            "javac7.srcjar",
        ],
        "//conditions:default": [
            "BUILD",
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
