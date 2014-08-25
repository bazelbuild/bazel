package(default_visibility = ["//visibility:public"])

cc_library(name = "malloc")

cc_library(name = "stl")

filegroup(
    name = "empty",
    srcs = [],
)

filegroup(
    name = "empty-local",
    srcs = [],
)

filegroup(
    name = "toolchain",
    srcs = [
        ":cc-compiler-local",
        ":compile-local",
        ":link-local",
        ":objcopy-local",
    ],
)

cc_toolchain(
    name = "cc-compiler-local",
    all_files = ":empty",
    compiler_files = ":empty",
    cpu = "local",
    dwp_files = ":empty",
    dynamic_runtime_libs = [":empty-local"],
    linker_files = ":empty",
    objcopy_files = ":empty",
    static_runtime_libs = [":empty-local"],
    strip_files = ":empty",
)
