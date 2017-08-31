package(default_visibility = ["//visibility:public"])

cc_library(
    name = "malloc",
)

cc_library(
    name = "stl",
)

filegroup(
    name = "empty",
    srcs = [],
)

filegroup(
    name = "cc_wrapper",
    srcs = ["cc_wrapper.sh"],
)

# This is the entry point for --crosstool_top.  Toolchains are found
# by lopping off the name of --crosstool_top and searching for
# the "${CPU}" entry in the toolchains attribute.
cc_toolchain_suite(
    name = "toolchain",
    toolchains = {
        "%{name}|%{compiler}": ":cc-compiler-%{name}",
        "armeabi-v7a|compiler": ":cc-compiler-armeabi-v7a",
        "ios_x86_64|compiler": ":cc-compiler-ios_x86_64",
    },
)

cc_toolchain(
    name = "cc-compiler-%{name}",
    all_files = "%{cc_compiler_deps}",
    compiler_files = "%{cc_compiler_deps}",
    cpu = "%{name}",
    dwp_files = ":empty",
    dynamic_runtime_libs = [":empty"],
    linker_files = "%{cc_compiler_deps}",
    objcopy_files = ":empty",
    static_runtime_libs = [":empty"],
    strip_files = ":empty",
    supports_param_files = %{supports_param_files},
)


# Android tooling requires a default toolchain for the armeabi-v7a cpu.
cc_toolchain(
    name = "cc-compiler-armeabi-v7a",
    all_files = ":empty",
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

# ios crosstool configuration requires a default toolchain for the
# ios_x86_64 cpu.
cc_toolchain(
    name = "cc-compiler-ios_x86_64",
    all_files = ":empty",
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
