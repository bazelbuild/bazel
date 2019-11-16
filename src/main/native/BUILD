load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

genrule(
    name = "copy_link_jni_md_header",
    srcs = select({
        "//src/conditions:darwin": ["//tools/jdk:jni_md_header-darwin"],
        "//src/conditions:darwin_x86_64": ["//tools/jdk:jni_md_header-darwin"],
        "//src/conditions:freebsd": ["//tools/jdk:jni_md_header-freebsd"],
        "//src/conditions:openbsd": ["//tools/jdk:jni_md_header-openbsd"],
        "//src/conditions:windows": ["//tools/jdk:jni_md_header-windows"],
        "//conditions:default": ["//tools/jdk:jni_md_header-linux"],
    }),
    outs = ["jni_md.h"],
    cmd = "cp -f $< $@",
    visibility = ["//src/main/native:__subpackages__"],
)

genrule(
    name = "copy_link_jni_header",
    srcs = ["//tools/jdk:jni_header"],
    outs = ["jni.h"],
    cmd = "cp -f $< $@",
    visibility = ["//src/main/native:__subpackages__"],
)

filegroup(
    name = "jni_os",
    srcs = select({
        "//src/conditions:darwin": [
            "unix_jni_darwin.cc",
            "fsevents.cc",
        ],
        "//src/conditions:darwin_x86_64": [
            "unix_jni_darwin.cc",
            "fsevents.cc",
        ],
        "//src/conditions:freebsd": ["unix_jni_freebsd.cc"],
        "//src/conditions:openbsd": ["unix_jni_openbsd.cc"],
        "//conditions:default": ["unix_jni_linux.cc"],
    }),
)

cc_library(
    name = "latin1_jni_path",
    srcs = [
        "latin1_jni_path.cc",
        ":jni.h",
        ":jni_md.h",
    ],
    hdrs = ["latin1_jni_path.h"],
    includes = ["."],  # For jni headers.
)

cc_binary(
    name = "libunix.so",
    srcs = [
        "macros.h",
        "process.cc",
        "unix_jni.cc",
        "unix_jni.h",
        ":jni.h",
        ":jni_md.h",
        ":jni_os",
    ],
    copts = [
        "-fPIC",
        "-DBLAZE_JAVA_CPU=\"k8\"",
    ],
    includes = ["."],  # For jni headers.
    linkopts = select({
        "//src/conditions:darwin": ["-framework CoreServices"],
        "//src/conditions:darwin_x86_64": ["-framework CoreServices"],
        "//conditions:default": [],
    }),
    linkshared = 1,
    visibility = ["//src:__subpackages__"],
    deps = [
        ":latin1_jni_path",
        "//src/main/cpp/util",
        "//src/main/cpp/util:md5",
    ],
)

# TODO(bazel-team): Come up with a way to support platform-specific dynamic
# library extensions.  This is issue #914.
genrule(
    name = "mac-compat",
    srcs = ["libunix.so"],
    outs = ["libunix.dylib"],
    cmd = "cp $< $@",
    output_to_bindir = 1,
    visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/main/native/windows:srcs",
    ],
    visibility = ["//src:__pkg__"],
)

filegroup(
    name = "embedded_tools",
    srcs = [
        "BUILD",
        "//src/main/native/windows:embedded_tools",
    ],
    visibility = ["//visibility:public"],
)
