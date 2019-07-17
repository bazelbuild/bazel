load(":win_res.bzl", "windows_resources")
load(":winsdk_toolchain.bzl", "windows_resource_compiler_toolchain")

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__pkg__"],
)

toolchain_type(
    name = "toolchain_type",
    visibility = ["//visibility:public"],
)

# ------------------------------------------------------------------------------
# Default toolchains. You can register these or define and register your own.
#
# To register these, add to the WORKSPACE file:
#
#     load("//src/main/res:local_config_winsdk.bzl", "local_config_winsdk")
#     local_config_winsdk(name = "local_config_winsdk")
#
#     register_toolchains(
#         "//src/main/res:win_local_rc_toolchain",
#         "//src/main/res:empty_rc_toolchain",
#     )

windows_resource_compiler_toolchain(
    name = "local",
    rc_exe = "@local_config_winsdk//:rc_exe.bat",
)

windows_resource_compiler_toolchain(
    name = "empty",
)

toolchain(
    name = "win_local_rc_toolchain",
    exec_compatible_with = [
        "@platforms//os:windows",
        "@platforms//cpu:x86_64",
    ],
    toolchain = ":local",
    toolchain_type = ":toolchain_type",
)

toolchain(
    name = "empty_rc_toolchain",
    toolchain = ":empty",
    toolchain_type = ":toolchain_type",
)
