package(default_visibility = ["//src:__subpackages__"])

cc_library(
    name = "logging",
    srcs = ["logging.cc"],
    hdrs = ["logging.h"],
)

cc_library(
    name = "process-tools",
    srcs = ["process-tools.cc"],
    hdrs = ["process-tools.h"],
    deps = [
        ":logging",
        "//src/main/protobuf:execution_statistics_cc_proto",
    ],
)

cc_binary(
    name = "process-wrapper",
    srcs = select({
        "//src:windows": ["process-wrapper-windows.cc"],
        "//src:windows_msvc": ["process-wrapper-windows.cc"],
        "//conditions:default": [
            "process-wrapper.cc",
            "process-wrapper.h",
            "process-wrapper-legacy.cc",
            "process-wrapper-legacy.h",
            "process-wrapper-options.cc",
            "process-wrapper-options.h",
        ],
    }),
    linkopts = ["-lm"],
    deps = select({
        "//src:windows": [],
        "//src:windows_msvc": [],
        "//conditions:default": [
            ":process-tools",
            ":logging",
        ],
    }),
)

cc_binary(
    name = "build-runfiles",
    srcs = select({
        "//src:windows": ["build-runfiles-windows.cc"],
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
        ],
    }),
    linkopts = ["-lm"],
    deps = select({
        "//src:darwin": [],
        "//src:darwin_x86_64": [],
        "//src:freebsd": [],
        "//src:windows": [],
        "//src:windows_msys": [],
        "//src:windows_msvc": [],
        "//conditions:default": [
            ":logging",
            ":process-tools",
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

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__pkg__"],
)
