package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

filegroup(
    name = "libtool_srcs",
    srcs = [
        "libtool.sh",
        "make_hashed_objlist.py",
    ],
)

sh_binary(
    name = "libtool",
    srcs = ["libtool.sh"],
    data = [
        ":make_hashed_objlist.py",
        "//src/tools/xcode/xcrunwrapper",
    ],
)
