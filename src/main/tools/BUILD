package(default_visibility = ["//src:__subpackages__"])

cc_binary(
    name = "daemonize",
    srcs = ["daemonize.cc"],
    deps = [
        ":process-tools",
    ],
)

cc_library(
    name = "logging",
    srcs = ["logging.cc"],
    hdrs = ["logging.h"],
)

cc_library(
    name = "process-tools",
    srcs = ["process-tools.cc"] + select({
        "//src/conditions:darwin": ["process-tools-darwin.cc"],
        "//src/conditions:openbsd": ["process-tools-darwin.cc"],
        "//conditions:default": ["process-tools-linux.cc"],
    }),
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
        "//src/conditions:darwin": ["-lm"],
        "//src/conditions:openbsd": [],
        "//src/conditions:windows": [],
        "//conditions:default": [
            "-lm",
            "-lrt",
        ],
    }),
    deps = select({
        "//src/conditions:windows": [],
        "//conditions:default": [
            ":logging",
            ":process-tools",
        ],
    }),
)

cc_binary(
    name = "build-runfiles",
    srcs = select({
        "//src/conditions:windows": ["build-runfiles-windows.cc"],
        "//conditions:default": ["build-runfiles.cc"],
    }),
    deps = ["//src/main/cpp/util:filesystem"] + select({
        "//src/conditions:windows": ["//src/main/native/windows:lib-file"],
        "//conditions:default": [],
    }),
)

cc_binary(
    name = "linux-sandbox",
    srcs = select({
        "//src/conditions:darwin": ["dummy-sandbox.c"],
        "//src/conditions:freebsd": ["dummy-sandbox.c"],
        "//src/conditions:openbsd": ["dummy-sandbox.c"],
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
    linkopts = select({
        "//src/conditions:darwin": [],
        "//src/conditions:freebsd": [],
        "//src/conditions:openbsd": [],
        "//src/conditions:windows": [],
        "//conditions:default": [
            "-lm",
            "-lrt",
        ],
    }),
    deps = select({
        "//src/conditions:darwin": [],
        "//src/conditions:freebsd": [],
        "//src/conditions:openbsd": [],
        "//src/conditions:windows": [],
        "//conditions:default": [
            ":logging",
            ":process-tools",
            "//src/main/cpp/util",
        ],
    }),
)

exports_files([
    "build_interface_so",
])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__pkg__"],
)
