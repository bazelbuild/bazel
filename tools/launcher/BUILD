package(default_visibility = ["//visibility:public"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

filegroup(
    name = "embedded_tools",
    srcs = [
        "BUILD.tools",
        "empty.sh",
    ],
    visibility = ["//tools:__pkg__"],
)

alias(
    name = "launcher",
    actual = "//src/tools/launcher:launcher",
)

alias(
    name = "launcher_maker",
    actual = "//src/tools/launcher:launcher_maker",
)
