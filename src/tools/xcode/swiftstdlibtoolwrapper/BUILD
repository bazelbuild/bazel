package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

# Sources that need to be packaged into embedded tools.
filegroup(
    name = "swiftstdlibtoolwrapper_srcs",
    srcs = [
        "swift_stdlib_tool.py",
        "swiftstdlibtoolwrapper.sh",
    ],
)

sh_binary(
    name = "swiftstdlibtoolwrapper",
    srcs = ["swiftstdlibtoolwrapper.sh"],
    data = [
        ":swift_stdlib_tool.py",
        "//src/tools/xcode/realpath",
        "//src/tools/xcode/xcrunwrapper",
    ],
)
