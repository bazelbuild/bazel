package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

sh_binary(
    name = "libtool",
    srcs = ["libtool.sh"],
    data = [
        "//src/tools/xcode/xcrunwrapper",
    ],
)
