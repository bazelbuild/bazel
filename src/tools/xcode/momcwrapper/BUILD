package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

sh_binary(
    name = "momcwrapper",
    srcs = ["momcwrapper.sh"],
    data = ["//src/tools/xcode/realpath:realpath_genrule"],
)
