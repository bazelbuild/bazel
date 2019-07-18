load(":winsdk_toolchain.bzl", "WINDOWS_RESOURCE_COMPILER_TOOLCHAIN_TYPE", "windows_resource_compiler_toolchain")

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__pkg__"],
)

# The toolchain type for every resource compiler toolchain.
#
# toolchain() rules must reference this by absolute label with repository
# name. See WINDOWS_RESOURCE_COMPILER_TOOLCHAIN_TYPE.
toolchain_type(
    name = "toolchain_type",
    visibility = ["//visibility:public"],
)

windows_resource_compiler_toolchain(
    name = "empty",
)

toolchain(
    name = "empty_rc_toolchain",
    toolchain = ":empty",
    toolchain_type = WINDOWS_RESOURCE_COMPILER_TOOLCHAIN_TYPE,
)
