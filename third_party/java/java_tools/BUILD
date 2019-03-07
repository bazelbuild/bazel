package(default_visibility = ["//visibility:public"])

licenses(["notice"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

filegroup(
    name = "java_tools",
    srcs = glob(
        ["*.jar"],
        exclude = ["bazel-singlejar_deploy.jar"],
    ) + select({
        "//src/conditions:arm": ["bazel-singlejar_deploy.jar"],
        "//conditions:default": [],
    }),
)
