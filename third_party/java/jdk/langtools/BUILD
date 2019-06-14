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
        "java_compiler.jar",
        "java_compiler-src.jar",
        "javac-9+181-r4173-1.jar",
        "javac-9+181-r4173-1.srcjar",
        "jdk_compiler.jar",
        "jdk_compiler-src.jar",
    ],
)

filegroup(
    name = "license-and-srcs",
    srcs = [
        "LICENSE",
        "java_compiler-src.jar",
        "javac-9+181-r4173-1.srcjar",
        "jdk_compiler-src.jar",
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

filegroup(
    name = "jdk_compiler_jar",
    srcs = ["jdk_compiler.jar"],
)

filegroup(
    name = "java_compiler_jar",
    srcs = ["java_compiler.jar"],
)
