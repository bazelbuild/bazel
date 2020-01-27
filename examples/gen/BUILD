package(default_visibility = ["//visibility:public"])

genquery(
    name = "genquery",
    expression = "deps(@bazel_tools//tools/jdk:current_java_runtime)",
    scope = ["@bazel_tools//tools/jdk:current_java_runtime"],
)

genrule(
    name = "genrule",
    srcs = [":genquery"],
    outs = ["genrule.txt"],
    cmd = "cat $(SRCS) > $@",
)

filegroup(
    name = "srcs",
    srcs = [
        "BUILD",
    ],
)
