package(default_visibility = ["//src:__subpackages__"])

cc_library(
    name = "network-tools",
    srcs = ["network-tools.c"],
    hdrs = ["network-tools.h"],
    copts = ["-std=c99"],
    deps = [":process-tools"],
)

cc_library(
    name = "process-tools",
    srcs = ["process-tools.c"],
    hdrs = ["process-tools.h"],
    copts = ["-std=c99"],
)

cc_binary(
    name = "process-wrapper",
    srcs = ["process-wrapper.c"],
    copts = ["-std=c99"],
    linkopts = ["-lm"],
    deps = [":process-tools"],
)

cc_binary(
    name = "build-runfiles",
    srcs = ["build-runfiles.cc"],
)

cc_binary(
    name = "namespace-sandbox",
    srcs = select({
        "//src:darwin": ["namespace-sandbox-dummy.c"],
        "//src:darwin_x86_64": ["namespace-sandbox-dummy.c"],
        "//src:freebsd": ["namespace-sandbox-dummy.c"],
        "//src:windows": ["namespace-sandbox-dummy.c"],
        "//conditions:default": ["namespace-sandbox.c"],
    }),
    copts = ["-std=c99"],
    linkopts = ["-lm"],
    deps = select({
        "//src:darwin": [],
        "//src:darwin_x86_64": [],
        "//src:freebsd": [],
        "//conditions:default": [
            ":process-tools",
            ":network-tools",
        ],
    }),
)

filegroup(
    name = "jdk-support",
    srcs = [
        "jdk.BUILD",
    ],
)

exports_files([
    "build_interface_so",
])
