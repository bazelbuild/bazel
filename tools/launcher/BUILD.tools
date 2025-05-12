load("@platforms//host:constraints.bzl", "HOST_CONSTRAINTS")
load(
    "//tools:build_defs.bzl",
    "IS_HOST_WINDOWS",
    "current_launcher_binary",
    "single_binary_toolchain",
)

package(default_visibility = ["//visibility:public"])

# WARNING: These targets and toolchain types only exist for the purposes of
# rulesets formerly included in Bazel itself and may change or be removed at any
# time.

current_launcher_binary(name = "launcher")

# DEPRECATED: Use the `:launcher_maker_toolchain_type` toolchain instead to
# avoid an unnecessary dependency on a C++ toolchain when building for a
# non-Windows platform.
filegroup(
    name = "launcher_maker",
    srcs = select({
        ":is_host": ["launcher_maker.exe" if IS_HOST_WINDOWS else "//src/tools/launcher:launcher_maker"],
        "//conditions:default": ["//src/tools/launcher:launcher_maker"],
    }),
)

config_setting(
    name = "is_host",
    constraint_values = HOST_CONSTRAINTS,
    visibility = ["//visibility:private"],
)

toolchain_type(name = "launcher_toolchain_type")

toolchain_type(name = "launcher_maker_toolchain_type")

# Toolchains are prefixed with a number to ensure that their order of definition
# matches their precedence in the toolchain resolution process when registered
# with a wildcard pattern (which sorts by name).
# TODO(#19587): Make all prebuilt binaries available in external repos and add
#  toolchains for them below so that cross-platform builds can use them without
#  needing to build from source.
IS_HOST_WINDOWS and single_binary_toolchain(
    name = "1_prebuilt_launcher",
    binary = "launcher.exe",
    target_compatible_with = HOST_CONSTRAINTS,
    toolchain_type = ":launcher_toolchain_type",
)

single_binary_toolchain(
    name = "2_source_launcher_toolchain",
    binary = "//src/tools/launcher",
    target_compatible_with = ["@platforms//os:windows"],
    toolchain_type = ":launcher_toolchain_type",
)

single_binary_toolchain(
    name = "3_no_launcher_toolchain",
    binary = "empty.sh",
    toolchain_type = ":launcher_toolchain_type",
)

IS_HOST_WINDOWS and single_binary_toolchain(
    name = "1_prebuilt_launcher_maker",
    binary = "launcher_maker.exe",
    exec_compatible_with = HOST_CONSTRAINTS,
    target_compatible_with = ["@platforms//os:windows"],
    toolchain_type = ":launcher_maker_toolchain_type",
)

single_binary_toolchain(
    name = "2_source_launcher_maker_toolchain",
    binary = "//src/tools/launcher:launcher_maker",
    target_compatible_with = ["@platforms//os:windows"],
    toolchain_type = ":launcher_maker_toolchain_type",
)

single_binary_toolchain(
    name = "3_no_launcher_maker_toolchain",
    binary = "empty.sh",
    toolchain_type = ":launcher_maker_toolchain_type",
)
