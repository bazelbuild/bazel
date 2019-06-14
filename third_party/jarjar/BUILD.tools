package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load(
    "//tools/jdk:remote_java_tools_aliases.bzl",
    "remote_java_tools_java_import",
)

remote_java_tools_java_import(
    name = "jarjar_import",
    target = ":JarJar",
)

java_binary(
    name = "jarjar_bin",
    main_class = "com.tonicsystems.jarjar.Main",
    runtime_deps = [":jarjar_import"],
)
