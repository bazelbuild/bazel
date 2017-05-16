package(default_visibility = ["//src:__subpackages__"])

cc_library(
    name = "process-tools",
    srcs = [
        "process-tools.cc",
        "process-tools.h",
    ],
)

cc_binary(
    name = "process-wrapper",
    srcs = select({
        "//src:windows_msvc": ["process-wrapper-windows.cc"],
        "//conditions:default": [
            "process-wrapper.cc",
        ],
    }),
    linkopts = ["-lm"],
    deps = select({
        "//src:windows_msvc": [],
        "//conditions:default": [
            ":process-tools",
        ],
    }),
)

cc_binary(
    name = "build-runfiles",
    srcs = select({
        "//src:windows_msvc": ["build-runfiles-windows.cc"],
        "//conditions:default": ["build-runfiles.cc"],
    }),
)

cc_binary(
    name = "linux-sandbox",
    srcs = select({
        "//src:darwin": ["dummy-sandbox.c"],
        "//src:darwin_x86_64": ["dummy-sandbox.c"],
        "//src:freebsd": ["dummy-sandbox.c"],
        "//src:windows": ["dummy-sandbox.c"],
        "//src:windows_msys": ["dummy-sandbox.c"],
        "//src:windows_msvc": ["dummy-sandbox.c"],
        "//conditions:default": [
            "linux-sandbox.cc",
            "linux-sandbox.h",
            "linux-sandbox-options.cc",
            "linux-sandbox-options.h",
            "linux-sandbox-pid1.cc",
            "linux-sandbox-pid1.h",
            "linux-sandbox-utils.h",
        ],
    }),
    linkopts = ["-lm"],
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

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__pkg__"],
)
