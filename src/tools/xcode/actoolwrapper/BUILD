package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

sh_binary(
    name = "actoolwrapper",
    srcs = ["actoolwrapper.sh"],
    data = [
        "//src/tools/xcode/realpath",
        "//src/tools/xcode/xcrunwrapper",
    ],
)
