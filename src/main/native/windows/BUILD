load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/native:__pkg__"],
)

filegroup(
    name = "embedded_tools",
    srcs = glob([
        "*.cc",
        "*.h",
    ]) + [
        "BUILD",
    ],
    visibility = ["//src/main/native:__pkg__"],
)

cc_library(
    name = "lib-file",
    srcs = [
        "file.cc",
        "util.cc",
    ],
    hdrs = [
        "file.h",
        "util.h",
    ],
    target_compatible_with = ["@platforms//os:windows"],
    visibility = [
        "//src/main/cpp:__subpackages__",
        "//src/main/tools:__pkg__",
        "//src/test/cpp:__subpackages__",
        "//src/test/native:__subpackages__",
        "//src/tools/launcher/util:__pkg__",
        "//tools/test:__pkg__",
    ],
)

cc_library(
    name = "lib-process",
    srcs = ["process.cc"],
    hdrs = ["process.h"],
    target_compatible_with = ["@platforms//os:windows"],
    visibility = [
        "//src/main/cpp:__pkg__",
        "//src/test/native:__subpackages__",
        "//src/tools/launcher:__pkg__",
        "//tools/test:__pkg__",
    ],
    deps = [":lib-file"],
)

cc_library(
    name = "lib-stat",
    srcs = ["stat.cc"],
    hdrs = ["stat.h"],
    target_compatible_with = ["@platforms//os:windows"],
    visibility = ["//src/test/native:__subpackages__"],
    deps = [":lib-file"],
)

cc_binary(
    name = "windows_jni.dll",
    srcs = [
        "file-jni.cc",
        "jni-util.cc",
        "jni-util.h",
        "processes-jni.cc",
        "//src/main/native:jni.h",
        "//src/main/native:jni_md.h",
    ],
    linkopts = select({
        # System libraries needed on Windows:
        #   kernel32: SetThreadExecutionState
        #   ntdll, synchronization: Rust libstd (passed directly because
        #       rules_rs v0.2.4 omits them from msvc stdlib_linkflags)
        "//src/conditions:windows_msvc_like": [
            "kernel32.lib",
            "ntdll.lib",
            "synchronization.lib",
        ],
        "//conditions:default": [],
    }),
    linkshared = 1,
    target_compatible_with = ["@platforms//os:windows"],
    visibility = [
        "//src/main/java/com/google/devtools/build/lib/jni:__pkg__",
    ],
    win_def_file = "windows_jni.def",
    deps = [
        ":lib-file",
        ":lib-process",
        ":lib-stat",
        "//src/main/native:blake3_jni",
        "//src/rust/platform:sleep_prevention_ffi",
    ],
)

filegroup(
    name = "embedded_java_tools",
    srcs = [
        "file.cc",
        "file.h",
        "util.cc",
        "util.h",
    ],
    visibility = ["//src/main/cpp/util:__pkg__"],
)

# A zip containing source files used by ijar.
genrule(
    name = "native_windows_zip",
    srcs = [":embedded_java_tools"],
    outs = ["native_windows.zip"],
    cmd = "$(location //src:zip_files) src/main/native/windows $@ $(SRCS)",
    tools = ["//src:zip_files"],
    visibility = ["//src/main/cpp/util:__pkg__"],
)
