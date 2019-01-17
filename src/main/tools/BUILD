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
        "//src/conditions:windows": ["process-wrapper-windows.cc"],
        "//conditions:default": [
            "process-wrapper.cc",
            "process-wrapper.h",
            "process-wrapper-legacy.cc",
            "process-wrapper-legacy.h",
            "process-wrapper-options.cc",
            "process-wrapper-options.h",
        ],
    }),
    linkopts = select({
        "//src/conditions:windows": [],
        "//conditions:default": ["-lm"],
    }),
    deps = select({
        "//src/conditions:windows": [],
        "//conditions:default": [
            ":process-tools",
            ":logging",
        ],
    }),
)

cc_binary(
    name = "build-runfiles",
    srcs = select({
        "//src/conditions:windows": ["build-runfiles-windows.cc"],
        "//conditions:default": ["build-runfiles.cc"],
    }),
    deps = ["//src/main/cpp/util:filesystem"],
)

cc_binary(
    name = "linux-sandbox",
    srcs = select({
        "//tools/osx:darwin": ["dummy-sandbox.c"],
        "//tools/osx:darwin_x86_64": ["dummy-sandbox.c"],
        "//src/conditions:freebsd": ["dummy-sandbox.c"],
        "//src/conditions:windows": ["dummy-sandbox.c"],
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
        "//tools/osx:darwin": [],
        "//tools/osx:darwin_x86_64": [],
        "//src/conditions:freebsd": [],
        "//src/conditions:windows": [],
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
