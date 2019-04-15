package(default_visibility = ["//visibility:private"])

load("//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

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
    srcs = ["file.cc"],
    hdrs = ["file.h"],
    visibility = [
        "//src/main/cpp:__subpackages__",
        "//src/main/tools:__pkg__",
        "//src/test/cpp:__subpackages__",
        "//src/test/native:__subpackages__",
        "//tools/test:__pkg__",
    ],
    deps = [":lib-util"],
)

cc_library(
    name = "lib-util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    visibility = [
        "//src/tools/launcher/util:__pkg__",
        "//tools/test:__pkg__",
    ],
)

cc_library(
    name = "lib-process",
    srcs = ["process.cc"],
    hdrs = ["process.h"],
    deps = [":lib-util"],
    visibility = ["//tools/test:__pkg__"],
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
    linkshared = 1,
    visibility = [
        "//src:__pkg__",
        "//src/test/java/com/google/devtools/build/lib:__subpackages__",
        "//src/tools/android/java/com/google/devtools/build/android:__subpackages__",
    ],
    deps = [
        ":lib-file",
        ":lib-process",
        ":lib-util",
    ],
)

# Directly re-export resources.o so it can be consumed in the client.
cc_library(
    name = "resources",
    srcs = ["resources.o"],
    linkstatic = 1,
    visibility = ["//src/main/cpp:__pkg__"],
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
