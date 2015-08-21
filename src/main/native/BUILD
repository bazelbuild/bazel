genrule(
    name = "copy_link_jni_md_header",
    srcs = select({
        "//src:darwin": ["//tools/jdk:jni_md_header-darwin"],
        "//conditions:default": ["//tools/jdk:jni_md_header-linux"],
    }),
    outs = ["jni_md.h"],
    cmd = "cp -f $< $@",
)

genrule(
    name = "copy_link_jni_header",
    srcs = ["//tools/jdk:jni_header"],
    outs = ["jni.h"],
    cmd = "cp -f $< $@",
)

filegroup(
    name = "jni_os",
    srcs = select({
        "//src:darwin": ["unix_jni_darwin.cc"],
        "//conditions:default": ["unix_jni_linux.cc"],
    }),
)

cc_binary(
    name = "libunix.so",
    srcs = [
        "localsocket.cc",
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
    linkshared = 1,
    visibility = ["//src:__subpackages__"],
    deps = [
        "//src/main/cpp/util",
        "//src/main/cpp/util:md5",
    ],
)

# HACK for Mac: copy libunix.so to libunix.dylib. We'll need to come up with a
# way to support platform-specific dynamic library extensions.
genrule(
    name = "mac-compat",
    srcs = ["libunix.so"],
    outs = ["libunix.dylib"],
    cmd = "cp $< $@",
    visibility = ["//src:__subpackages__"],
)
