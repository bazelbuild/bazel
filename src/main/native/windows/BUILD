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

# A zip containing source files used by ijar.
genrule(
    name = "native_windows_zip",
    srcs = [
        "file.cc",
        "file.h",
        "util.cc",
        "util.h",
    ],
    outs = ["native_windows.zip"],
    cmd = "$(location //third_party:zip_files.sh) src/main/native/windows $(OUTS) $(SRCS)",
    tools = ["//third_party:zip_files.sh"],
    visibility = ["//src/main/cpp/util:__pkg__"],
)
