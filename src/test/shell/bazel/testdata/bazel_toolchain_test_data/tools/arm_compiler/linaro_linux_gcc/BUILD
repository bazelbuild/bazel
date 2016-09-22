package(default_visibility = ["//tools/arm_compiler:__pkg__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/test/shell/bazel/testdata/bazel_toolchain_test_data/tools/arm_compiler:__pkg__"],
)

filegroup(
    name = "gcc",
    srcs = [
        "arm-linux-gnueabihf-gcc",
        "@linaroLinuxGcc49Repo//:gcc",
    ],
)

filegroup(
    name = "ar",
    srcs = [
        "arm-linux-gnueabihf-ar",
        "@linaroLinuxGcc49Repo//:ar",
    ],
)

filegroup(
    name = "ld",
    srcs = [
        "arm-linux-gnueabihf-ld",
        "@linaroLinuxGcc49Repo//:ld",
    ],
)

filegroup(
    name = "nm",
    srcs = [
        "arm-linux-gnueabihf-nm",
        "@linaroLinuxGcc49Repo//:nm",
    ],
)

filegroup(
    name = "objcopy",
    srcs = [
        "arm-linux-gnueabihf-objcopy",
        "@linaroLinuxGcc49Repo//:objcopy",
    ],
)

filegroup(
    name = "objdump",
    srcs = [
        "arm-linux-gnueabihf-objdump",
        "@linaroLinuxGcc49Repo//:objdump",
    ],
)

filegroup(
    name = "strip",
    srcs = [
        "arm-linux-gnueabihf-strip",
        "@linaroLinuxGcc49Repo//:strip",
    ],
)

filegroup(
    name = "as",
    srcs = [
        "arm-linux-gnueabihf-as",
        "@linaroLinuxGcc49Repo//:as",
    ],
)

filegroup(
    name = "tool-wrappers",
    srcs = [
        ":ar",
        ":as",
        ":gcc",
        ":ld",
        ":nm",
        ":objcopy",
        ":objdump",
        ":strip",
    ],
)
