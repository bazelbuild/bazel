package(default_visibility = ["//visibility:public"])

licenses(["notice"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

filegroup(
    name = "bazel_embedded_java_tools",
    srcs = [
        "BUILD.pkg",
        "BUILD-new.pkg",
    ],
)

filegroup(
    name = "java_tools",
    srcs = glob(
        ["*.jar"],
        exclude = ["bazel-singlejar_deploy.jar"],
    ) + select({
        "//src/conditions:arm": ["bazel-singlejar_deploy.jar"],
        "//conditions:default": [],
    }) + [
        "BUILD.pkg",
        "BUILD-new.pkg",
    ],
)
