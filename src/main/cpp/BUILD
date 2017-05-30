# Description:
#   The Bazel launcher.
package(
    default_visibility = ["//visibility:public"],
)

WIN_LINK_OPTS = [
    "-Wl,advapi32.lib",  # GetUserNameW
    "-Wl,ole32.lib",  # CoTaskMemFree
    "-Wl,shell32.lib",  # SHGetKnownFolderPath
    "-Wl,ws2_32.lib",  # grpc
]

cc_library(
    name = "blaze_util",
    srcs = [
        "blaze_util.cc",
        "global_variables.h",
        "startup_options.h",
    ] + select({
        "//src:darwin": [
            "blaze_util_darwin.cc",
            "blaze_util_posix.cc",
        ],
        "//src:darwin_x86_64": [
            "blaze_util_darwin.cc",
            "blaze_util_posix.cc",
        ],
        "//src:freebsd": [
            "blaze_util_freebsd.cc",
            "blaze_util_posix.cc",
        ],
        "//src:windows": [
            "blaze_util_windows.cc",
        ],
        "//src:windows_msys": [
            "blaze_util_windows.cc",
        ],
        "//src:windows_msvc": [
            "blaze_util_windows.cc",
        ],
        "//conditions:default": [
            "blaze_util_linux.cc",
            "blaze_util_posix.cc",
        ],
    }),
    hdrs = [
        "blaze_util.h",
        "blaze_util_platform.h",
    ],
    linkopts = select({
        "//src:darwin": [
            "-framework CoreFoundation",
        ],
        "//src:darwin_x86_64": [
            "-framework CoreFoundation",
        ],
        "//src:freebsd": [
        ],
        "//src:windows": WIN_LINK_OPTS,
        "//src:windows_msvc": WIN_LINK_OPTS,
        "//conditions:default": [
            "-lrt",
        ],
    }),
    deps = [
        "//src/main/cpp/util",
        "//src/main/cpp/util:blaze_exit_code",
    ] + select({
        "//src:windows": ["//src/main/native:windows_jni_lib"],
        "//src:windows_msys": ["//src/main/native:windows_jni_lib"],
        "//src:windows_msvc": ["//src/main/native:windows_jni_lib"],
        "//conditions:default": [],
    }),
)

cc_binary(
    name = "client",
    srcs = [
        "blaze.cc",
        "blaze.h",
        "global_variables.cc",
        "global_variables.h",
        "main.cc",
        "option_processor.cc",
        "option_processor.h",
        "startup_options.cc",
        "startup_options.h",
        "workspace_layout.cc",
        "workspace_layout.h",
    ],
    copts = [
        "-Wno-sign-compare",
    ],
    linkopts = select({
        "//src:darwin": [
        ],
        "//src:darwin_x86_64": [
        ],
        "//src:freebsd": [
            "-lprocstat",
            "-lm",
        ],
        "//src:windows_msvc": [
        ],
        "//conditions:default": [
            "-lrt",
            "-ldl",
        ],
    }),
    visibility = ["//src:__pkg__"],
    deps = [
        ":blaze_util",
        "//src/main/cpp/util",
        "//src/main/cpp/util:errors",
        "//src/main/cpp/util:logging",
        "//src/main/cpp/util:strings",
        "//src/main/protobuf:command_server_cc_proto",
        "//third_party/ijar:zip",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + ["//src/main/cpp/util:srcs"],
    visibility = ["//src:__pkg__"],
)
