load("@bazel_tools//tools/jdk:default_java_toolchain.bzl", "java_runtime_files")

genrule(
    name = "copy_jni_header",
    srcs = ["@bazel_tools//tools/jdk:jni_header"],
    outs = ["jni/jni.h"],
    cmd = "cp -f $< $@",
)

genrule(
    name = "copy_jni_md_header",
    srcs = select({
        "@bazel_tools//src/conditions:darwin": ["@bazel_tools//tools/jdk:jni_md_header-darwin"],
        "@bazel_tools//src/conditions:freebsd": ["@bazel_tools//tools/jdk:jni_md_header-freebsd"],
        "@bazel_tools//src/conditions:openbsd": ["@bazel_tools//tools/jdk:jni_md_header-openbsd"],
        "@bazel_tools//src/conditions:windows": ["@bazel_tools//tools/jdk:jni_md_header-windows"],
        "//conditions:default": ["@bazel_tools//tools/jdk:jni_md_header-linux"],
    }),
    outs = ["jni/jni_md.h"],
    cmd = "cp -f $< $@",
)

java_runtime_files(
    name = "jvmti_header",
    srcs = ["include/jvmti.h"],
)

genrule(
    name = "copy_jvmti_header",
    srcs = ["include/jvmti.h"],
    outs = ["jni/jvmti.h"],
    cmd = "cp -f $< $@",
)

# Keep in sync with the build/libasyncProfiler.so Makefile target.
filegroup(
    name = "source_files",
    srcs = glob([
        "src/*.cpp",
        "src/*.h",
        "src/fdtransfer/*.h",
    ]),
)

# These files are inlined by the .incbin assembler directive.
# We can't put them in cc_binary.srcs, so we rename them to .inl.
# This works in conjunction with incbin.patch.
INLINED_FILES = [
    "src/res/flame.html",
    "src/res/tree.html",
    "src/helper/one/profiler/Instrument.class",
    "src/helper/one/profiler/JfrSync.class",
    "src/helper/one/profiler/Server.class",
]

genrule(
    name = "inlined_files",
    srcs = INLINED_FILES,
    outs = [f + ".inl" for f in INLINED_FILES],
    cmd = " && ".join([
        "mv $(location {f}) $(location {f}.inl)".format(f=f)
        for f in INLINED_FILES
     ]),
)

cc_binary(
    name = "libasync_profiler.so",
    srcs = [
        ":source_files",
        ":inlined_files",
        ":jni/jni.h",
        ":jni/jni_md.h",
        ":jni/jvmti.h",
    ],
    includes = [
        "jni",
    ],
    copts = [
      # Search path for files inlined by the .incbin assembler directive.
      # The includes attribute doesn't seem to work for these.
      "-Wa,-I$(BINDIR)/external/async-profiler/src/res",
      "-Wa,-I$(BINDIR)/external/async-profiler/src/helper",
    ],
    local_defines = [
        # Keep in sync with Makefile.
        "PROFILER_VERSION=\\\"2.9\\\"",
    ] + select({
        "@bazel_tools//src/conditions:darwin": [
            # For <ucontext.h>.
            "_XOPEN_SOURCE=1",
        ],
        "//conditions:default": [],
    }),
    linkshared = 1,
)

genrule(
    name = "libasync_profiler_macos",
    srcs = ["libasync_profiler.so"],
    outs = ["libasync_profiler.dylib"],
    cmd = "cp $< $@",
)

filegroup(
    name = "async_profiler_jni",
    srcs = select({
        "@bazel_tools//src/conditions:linux": [":libasync_profiler.so"],
        "@bazel_tools//src/conditions:darwin": [":libasync_profiler.dylib"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

java_library(
    name = "async_profiler",
    srcs = glob([
        "src/api/one/profiler/*.java",
        "src/converter/**/*.java",
    ]),
    visibility = ["//visibility:public"],
)
