# This is the entry point for --crosstool_top.  Toolchains are found
# by lopping off the name of --crosstool_top and searching for
# 'cc-compiler-${CPU}' in this BUILD file, where CPU is the target CPU
# specified in --cpu.
#
# This file group should include
#   * all cc_toolchain targets supported
#   * all file groups that said cc_toolchain might refer to,
# including the default_grte_top setting in the CROSSTOOL
# protobuf.
filegroup(
    name = "toolchain_fg",
    srcs = [
        ":cc-compiler-armeabi-v7a",
        ":cc-compiler-k8",
        ":linaro_linux_all_files",
        "@org_linaro_components_toolchain_gcc_5_3_1//:compiler_components",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/test/shell/bazel/testdata/bazel_toolchain_test_data/tools/arm_compiler/linaro_linux_gcc:srcs",
    ],
    visibility = ["//src/test/shell/bazel/testdata/bazel_toolchain_test_data:__pkg__"],
)

cc_toolchain_suite(
    name = "toolchain",
    # target_cpu | compiler
    toolchains = {
        "armeabi-v7a": "cc-compiler-armeabi-v7a",
        "k8": "cc-compiler-k8",
        "armeabi-v7a|gcc": "cc-compiler-armeabi-v7a",
        "k8|compiler": "cc-compiler-k8",
    },
    visibility = ["//visibility:public"],
)

filegroup(
    name = "linaro_linux_all_files",
    srcs = [
        "//tools/arm_compiler/linaro_linux_gcc:tool-wrappers",
        "@org_linaro_components_toolchain_gcc_5_3_1//:compiler_pieces",
    ],
)

filegroup(
    name = "linaro_linux_linker_files",
    srcs = [
        "//tools/arm_compiler/linaro_linux_gcc:ar",
        "//tools/arm_compiler/linaro_linux_gcc:gcc",
        "//tools/arm_compiler/linaro_linux_gcc:ld",
        "@org_linaro_components_toolchain_gcc_5_3_1//:compiler_pieces",
    ],
)

filegroup(
    name = "linaro_linux_compiler_files",
    srcs = [
        "//tools/arm_compiler/linaro_linux_gcc:as",
        "//tools/arm_compiler/linaro_linux_gcc:gcc",
        "//tools/arm_compiler/linaro_linux_gcc:ld",
    ],
)

filegroup(
    name = "empty",
    srcs = [],
)

cc_toolchain(
    name = "cc-compiler-armeabi-v7a",
    toolchain_identifier = "armeabi-v7a",
    all_files = ":linaro_linux_all_files",
    ar_files = "//tools/arm_compiler/linaro_linux_gcc:ar",
    as_files = "//tools/arm_compiler/linaro_linux_gcc:as",
    compiler_files = ":linaro_linux_compiler_files",
    cpu = "armeabi-v7a",
    dwp_files = ":empty",
    dynamic_runtime_libs = [":empty"],
    linker_files = ":linaro_linux_linker_files",
    objcopy_files = "//tools/arm_compiler/linaro_linux_gcc:objcopy",
    static_runtime_libs = [":empty"],
    strip_files = "//tools/arm_compiler/linaro_linux_gcc:strip",
    supports_param_files = 1,
    visibility = ["//visibility:public"],
)

cc_toolchain(
    name = "cc-compiler-k8",
    toolchain_identifier = "local",
    all_files = ":empty",
    ar_files = ":empty",
    as_files = ":empty",
    compiler_files = ":empty",
    cpu = "local",
    dwp_files = ":empty",
    dynamic_runtime_libs = [":empty"],
    linker_files = ":empty",
    objcopy_files = ":empty",
    static_runtime_libs = [":empty"],
    strip_files = ":empty",
    supports_param_files = 1,
)
