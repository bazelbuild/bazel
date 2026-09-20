load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

# Description:
#   The Bazel launcher.
package(
    default_visibility = ["//visibility:public"],
)

# System libraries needed on Windows:
#   advapi32: GetUserNameW
#   ole32:    CoTaskMemFree
#   shell32:  SHGetKnownFolderPath
#   uuid:     FOLDERID_Profile (MinGW only, MSVC defines it inline)
#   ws2_32:   grpc
WIN_LINK_OPTS = select({
    "//src/conditions:windows_msvc_like": [
        "-DEFAULTLIB:advapi32.lib",
        "-DEFAULTLIB:ole32.lib",
        "-DEFAULTLIB:shell32.lib",
        "-DEFAULTLIB:ws2_32.lib",
    ],
    "//src/conditions:windows_gcc_like": [
        "-ladvapi32",
        "-lole32",
        "-lshell32",
        "-luuid",
        "-lws2_32",
    ],
    "//conditions:default": [],
})

cc_library(
    name = "blaze_util",
    srcs = [
        "blaze_util.cc",
        "server_process_info.h",
        "startup_options.h",
    ] + select({
        "//src/conditions:darwin": [
            "blaze_util_darwin.cc",
            "blaze_util_posix.cc",
        ],
        "//src/conditions:freebsd": [
            "blaze_util_bsd.cc",
            "blaze_util_posix.cc",
        ],
        "//src/conditions:openbsd": [
            "blaze_util_bsd.cc",
            "blaze_util_posix.cc",
        ],
        "//src/conditions:windows": [
            "blaze_util_windows.cc",
        ],
        "//conditions:default": [
            "blaze_util_linux.cc",
            "blaze_util_posix.cc",
            "get_self_path_linux.cc",
        ],
    }),
    hdrs = [
        "blaze_util.h",
        "blaze_util_platform.h",
    ],
    linkopts = select({
        "//src/conditions:darwin": [
            "-framework CoreFoundation",
        ],
        "//src/conditions:freebsd": [
        ],
        "//src/conditions:openbsd": [
        ],
        "//src/conditions:windows": [
        ],
        "//conditions:default": [
            "-lrt",
        ],
    }) + WIN_LINK_OPTS,
    deps = [
        "//src/main/cpp/util:blaze_exit_code",
        "//src/main/cpp/util:errors",
        "//src/main/cpp/util:filesystem",
        "//src/main/cpp/util:logging",
        "//src/main/cpp/util:md5",
        "//src/main/cpp/util:numbers",
        "//src/main/cpp/util:port",
        "//src/main/cpp/util:strings",
        "@abseil-cpp//absl/base:log_severity",
        "@abseil-cpp//absl/log:globals",
        "@abseil-cpp//absl/log:initialize",
        "@abseil-cpp//absl/time",
    ] + select({
        "//src/conditions:windows": [
            "//src/main/native/windows:lib-file",
            "//src/main/native/windows:lib-process",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "archive_utils",
    srcs = [
        "archive_utils.cc",
    ],
    hdrs = [
        "archive_utils.h",
    ],
    deps = [
        ":blaze_util",
        "//src/main/cpp/util:blaze_exit_code",
        "//src/main/cpp/util:errors",
        "//src/main/cpp/util:filesystem",
        "//src/main/cpp/util:logging",
        "//third_party/ijar:zip",
    ],
)

cc_binary(
    name = "client",
    srcs = [
        "blaze.cc",
        "blaze.h",
        "main.cc",
        "server_process_info.cc",
        "server_process_info.h",
    ] + select({
        "//src/conditions:windows": ["resources.o"],
        "//conditions:default": [],
    }),
    copts = select({
        "//src/conditions:windows_msvc_like": ["/wd4018"],
        "//conditions:default": ["-Wno-sign-compare"],
    }),
    linkopts = select({
        "//src/conditions:darwin": [
        ],
        "//src/conditions:freebsd": [
            "-lprocstat",
            "-lm",
        ],
        "//src/conditions:openbsd": [
        ],
        "//src/conditions:windows": [
        ],
        "//conditions:default": [
            "-lrt",
            "-ldl",
        ],
    }) + select({
        # MinGW requires -municode when using wmain.
        "//src/conditions:windows_gcc_like": [
            "-municode",
            # grpc and c-ares only pass -defaultlib:ws2_32.lib and abseil only
            # detects MinGW GCC, not clang, so their system library
            # dependencies are missing.
            # TODO: Remove once https://github.com/grpc/grpc/pull/43483,
            # https://github.com/bazelbuild/bazel-central-registry/pull/10644
            # and https://github.com/abseil/abseil-cpp/pull/2172
            # are available in the versions of grpc, c-ares and abseil used by
            # Bazel.
            "-lbcrypt",
            "-liphlpapi",
            "-lws2_32",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//src:__pkg__"],
    deps = [
        ":archive_utils",
        ":bazel_startup_options",
        ":blaze_util",
        ":command_extension_adder",
        ":option_processor",
        ":startup_interceptor",
        ":startup_options",
        ":workspace_layout",
        "//src/main/cpp/util:bazel_log_handler",
        "//src/main/cpp/util:blaze_exit_code",
        "//src/main/cpp/util:errors",
        "//src/main/cpp/util:filesystem",
        "//src/main/cpp/util:logging",
        "//src/main/cpp/util:numbers",
        "//src/main/cpp/util:port",
        "//src/main/cpp/util:strings",
        "//src/main/protobuf:command_server_cc_grpc",
        "//src/main/protobuf:command_server_cc_proto",
        "//third_party/ijar:zip",
        "@abseil-cpp//absl/time",
    ],
)

cc_library(
    name = "option_processor",
    srcs = [
        "option_processor.cc",
    ] + select({
        "//src/conditions:windows": ["option_processor_windows.cc"],
        "//conditions:default": ["option_processor_unix.cc"],
    }),
    hdrs = [
        "option_processor.h",
        "option_processor-internal.h",
    ],
    # The system bazelrc can be voided by setting BAZEL_SYSTEM_BAZELRC_PATH to
    # /dev/null.
    copts = select({
        # For windows platforms, this can include environment
        # variables in the form %ProgramData%. We need to escape for
        # multiple levels, here, this becomes
        # /DBAZEL_SYSTEM_BAZELRC_PATH="%%ProgramData%%/bazel.bazelrc"',
        # and the double % get reduced down to 1 by the compiler. A
        # forward slash is used because \b is a special character,
        # backspace.
        "//src/conditions:windows_msvc_like": [
            "/DBAZEL_SYSTEM_BAZELRC_PATH#\\\"%%ProgramData%%/bazel.bazelrc\\\"",
        ],
        "@platforms//os:windows": [
            "-DBAZEL_SYSTEM_BAZELRC_PATH=\\\"%%ProgramData%%/bazel.bazelrc\\\"",
        ],
        # For posix platforms, this can include environment variables in the
        # form ${var_name}. Braces are required.
        "//conditions:default": [
            "-DBAZEL_SYSTEM_BAZELRC_PATH=\\\"/etc/bazel.bazelrc\\\"",
        ],
    }),
    visibility = [
        "//src:__pkg__",
        "//src/test/cpp:__pkg__",
    ],
    deps = [
        ":blaze_util",
        ":rc_file",
        ":sem_ver",
        ":startup_options",
        ":workspace_layout",
        "//src/main/cpp/util:blaze_exit_code",
        "//src/main/cpp/util:filesystem",
        "//src/main/cpp/util:logging",
        "//src/main/cpp/util:strings",
        "@abseil-cpp//absl/strings",
    ],
)

genrule(
    name = "jvm_module_options_h",
    srcs = [
        "generate_jvm_module_options.sh",
        "//src/main/java/com/google/devtools/build/lib/bazel:BazelServer_deploy.jar",
    ],
    outs = ["jvm_module_options.h"],
    cmd = "$(location generate_jvm_module_options.sh) $(location //src/main/java/com/google/devtools/build/lib/bazel:BazelServer_deploy.jar) $(OUTS)",
)

cc_library(
    name = "jvm_module_options",
    hdrs = [":jvm_module_options_h"],
    includes = ["."],
)

cc_library(
    name = "startup_options",
    srcs = ["startup_options.cc"],
    hdrs = ["startup_options.h"],
    visibility = [
        "//src:__pkg__",
        "//src/test/cpp:__pkg__",
    ],
    deps = [
        ":blaze_util",
        ":jvm_module_options",
        ":workspace_layout",
        "//src/main/cpp/util:blaze_exit_code",
        "//src/main/cpp/util:errors",
        "//src/main/cpp/util:filesystem",
        "//src/main/cpp/util:logging",
        "//src/main/cpp/util:numbers",
        "//src/main/cpp/util:strings",
        "@abseil-cpp//absl/time",
    ],
)

cc_library(
    name = "bazel_startup_options",
    srcs = ["bazel_startup_options.cc"],
    hdrs = ["bazel_startup_options.h"],
    visibility = [
        "//src:__pkg__",
        "//src/test/cpp:__pkg__",
    ],
    deps = [
        ":blaze_util",
        ":startup_options",
        ":workspace_layout",
        "//src/main/cpp/util:blaze_exit_code",
        "//src/main/cpp/util:filesystem",
        "//src/main/cpp/util:logging",
    ],
)

cc_library(
    name = "workspace_layout",
    srcs = ["workspace_layout.cc"],
    hdrs = ["workspace_layout.h"],
    visibility = [
        "//src:__pkg__",
        "//src/test/cpp:__pkg__",
    ],
    deps = [
        ":blaze_util",
        "//src/main/cpp/util:filesystem",
    ],
)

cc_library(
    name = "rc_file",
    srcs = ["rc_file.cc"],
    hdrs = ["rc_file.h"],
    visibility = [
        "//src:__pkg__",
        "//src/test/cpp:__pkg__",
    ],
    deps = [
        ":blaze_util",
        ":sem_ver",
        ":workspace_layout",
        "//src/main/cpp/util:filesystem",
        "//src/main/cpp/util:logging",
        "//src/main/cpp/util:strings",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/functional:function_ref",
        "@abseil-cpp//absl/memory",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/types:span",
        "@re2",
    ],
)

cc_library(
    name = "sem_ver",
    srcs = ["sem_ver.cc"],
    hdrs = ["sem_ver.h"],
    visibility = [
        "//src:__pkg__",
        "//src/test/cpp:__pkg__",
    ],
    deps = [
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
        "@re2",
    ],
)

cc_library(
    name = "startup_interceptor",
    hdrs = ["startup_interceptor.h"],
    visibility = [
        "//src:__pkg__",
        "//src/test/cpp:__pkg__",
    ],
)

cc_library(
    name = "command_extension_adder",
    hdrs = ["command_extension_adder.h"],
    deps = ["//src/main/protobuf:command_server_cc_proto"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + ["//src/main/cpp/util:srcs"],
    visibility = ["//src:__pkg__"],
)
