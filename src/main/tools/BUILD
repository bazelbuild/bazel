package(default_visibility = ["//src:__subpackages__"])

cc_binary(
    name = "process-wrapper",
    srcs = select({
        "//src:windows_msvc": ["process-wrapper-windows.cc"],
        "//conditions:default": [
            "process-tools.c",
            "process-tools.h",
            "process-wrapper.c",
        ],
    }),
    copts = select({
        "//src:windows_msvc": [],
        "//conditions:default": ["-std=c99"],
    }),
    linkopts = ["-lm"],
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
