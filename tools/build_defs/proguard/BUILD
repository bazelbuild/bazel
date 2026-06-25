load("@rules_java//java:defs.bzl", "java_binary")
load("@rules_python//python:py_binary.bzl", "py_binary")

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//tools:__subpackages__"],
)

exports_files(
    ["proguard.bzl"],
    visibility = ["//visibility:public"],
)

# Utility used to run proguard and fix timestamps.
py_binary(
    name = "wrapper_private",
    srcs = ["wrapper.py"],
    data = [
        ":proguard_private",
    ],
    main = "wrapper.py",
    visibility = ["//visibility:private"],
    deps = [
        "@rules_python//python/runfiles",
    ],
)

java_binary(
    name = "proguard_private",
    jvm_flags = [
        # Prevent ProGuard from calling out to the internet through log4j.
        "-Dlog4j.rootLogger=OFF",
    ],
    main_class = "proguard.ProGuard",
    visibility = ["//visibility:private"],
    runtime_deps = ["@maven//:com_guardsquare_proguard_base"],
)
