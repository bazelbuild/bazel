package(default_visibility = ["//src:__subpackages__"])

cc_binary(
    name = "process-wrapper",
    srcs = ["process-wrapper.c"],
    copts = ["-std=c99"],
)

cc_binary(
    name = "build-runfiles",
    srcs = ["build-runfiles.cc"],
)

cc_binary(
    name = "namespace-sandbox",
    srcs = select({
        "//src:darwin": ["namespace-sandbox-dummy.c"],
        "//conditions:default": ["namespace-sandbox.c"],
    }),
    copts = ["-std=c99"],
)

exports_files(["build_interface_so"])
