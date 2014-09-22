package(default_visibility = ["//visibility:public"])

# TODO(bazel-team): The MacOS tools give an error if there is no object file on
# the command line to libtool, so we use an empty .cc file here. We should find
# a better way to handle this case.
cc_library(
    name = "malloc",
    srcs = ["empty.cc"],
)

cc_library(
    name = "stl",
    srcs = ["empty.cc"],
)

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
    supports_param_files = 0,
)
