package(default_visibility = ["//visibility:public"])

sh_binary(
    name = "2to3",
    srcs = ["2to3.sh"],
)

filegroup(
    name = "srcs_and_embedded_tools",
    srcs = [
        # Tools are build from the workspace for tests.
        "2to3.sh",
        "BUILD",
    ],
    visibility = ["//visibility:private"],
)

filegroup(
    name = "srcs",
    srcs = [
        ":srcs_and_embedded_tools",
        "//tools/python/runfiles:srcs",
    ],
)

filegroup(
    name = "embedded_tools",
    srcs = [
        ":srcs_and_embedded_tools",
        "//tools/python/runfiles:embedded_tools",
    ],
    visibility = ["//tools:__pkg__"],
)
