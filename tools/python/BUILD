package(default_visibility = ["//visibility:public"])

sh_binary(
    name = "2to3",
    srcs = ["2to3.sh"],
)

filegroup(
    name = "srcs",
    srcs = [
        # Tools are build from the workspace for tests.
        "2to3.sh",
        "BUILD",
    ],
)
