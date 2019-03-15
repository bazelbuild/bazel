package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load(
    "//tools/jdk:remote_java_tools_aliases.bzl",
    "remote_java_tools_filegroup",
)


remote_java_tools_filegroup(
    name = "javac_jar",
    target = ":java_tools/javac-9+181-r4173-1.jar",
)

remote_java_tools_filegroup(
    name = "jdk_compiler_jar",
    target = ":java_tools/jdk_compiler.jar",
)

remote_java_tools_filegroup(
    name = "java_compiler_jar",
    target = ":java_tools/java_compiler.jar",
)
