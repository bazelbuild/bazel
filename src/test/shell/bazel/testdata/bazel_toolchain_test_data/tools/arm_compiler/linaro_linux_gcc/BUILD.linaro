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
        "@org_linaro_components_toolchain_gcc_5_3_1//:gcc",
    ],
)

filegroup(
    name = "ar",
    srcs = [
        "arm-linux-gnueabihf-ar",
        "@org_linaro_components_toolchain_gcc_5_3_1//:ar",
    ],
)

filegroup(
    name = "ld",
    srcs = [
        "arm-linux-gnueabihf-ld",
        "@org_linaro_components_toolchain_gcc_5_3_1//:ld",
    ],
)

filegroup(
    name = "nm",
    srcs = [
        "arm-linux-gnueabihf-nm",
        "@org_linaro_components_toolchain_gcc_5_3_1//:nm",
    ],
)

filegroup(
    name = "objcopy",
    srcs = [
        "arm-linux-gnueabihf-objcopy",
        "@org_linaro_components_toolchain_gcc_5_3_1//:objcopy",
    ],
)

filegroup(
    name = "objdump",
    srcs = [
        "arm-linux-gnueabihf-objdump",
        "@org_linaro_components_toolchain_gcc_5_3_1//:objdump",
    ],
)

filegroup(
    name = "strip",
    srcs = [
        "arm-linux-gnueabihf-strip",
        "@org_linaro_components_toolchain_gcc_5_3_1//:strip",
    ],
)

filegroup(
    name = "as",
    srcs = [
        "arm-linux-gnueabihf-as",
        "@org_linaro_components_toolchain_gcc_5_3_1//:as",
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
