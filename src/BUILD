# This is a quick and dirty way to compile Bazel using only genrule.
# We will need proper proto and java rules.

load('rules/genproto', 'genproto')
load('rules/java_library_oss', 'java_library')
load('rules/java_binary_oss', 'java_binary')

# The file assumes there is a directory (or symlink) src/javadir.
# Use this command to use local java:
#   ln -s $(which java | sed 's_/bin/java'__) src/javadir

JAVA_HOME = "src/javadir"

# Proto

[genproto(
    name = "proto_" + proto_file,
    src = "main/protobuf/" + proto_file + ".proto",
    ) for proto_file in ["build", "crosstool_config", "extra_actions_base",
                         "test_status", "bundlemerge", "xcodegen"]]

# Java

# TODO(bazel-team): Split this into multiple rules.
java_library(
    name = "xcode-common",
    srcs = glob([
        "tools/xcode-common/java/com/google/devtools/build/xcode/util/*.java",
        "tools/xcode-common/java/com/google/devtools/build/xcode/common/*.java",
    ]),
    jars = [
        "//third_party"
    ],
)

java_library(
    name = "blaze-core",
    srcs = glob([
        "main/**/*.java",
    ]),
    deps = [
        ":xcode-common",
        ":proto_build",
        ":proto_bundlemerge",
        ":proto_crosstool_config",
        ":proto_extra_actions_base",
        ":proto_test_status",
        ":proto_xcodegen",
    ],
    jars = [
        "//third_party",
    ],
)

java_library(
    name = "tests",
    srcs = glob([
        "test/**/*.java",
    ]),
    deps = [
        ":blaze-core",
        ":xcode-common",
        ":proto_build",
        ":proto_bundlemerge",
        ":proto_crosstool_config",
        ":proto_extra_actions_base",
        ":proto_test_status",
        ":proto_xcodegen",
    ],
    jars = [
        "//third_party",
        "//third_party:third_party_test_utils",
    ],
)

java_binary(
    name = "blaze",
    main_class = "com.google.devtools.build.lib.bazel.BazelMain",
    deps = [
        ":blaze-core",
    ],
)

# C++

cc_binary(
    name = "process-wrapper",
    srcs = ["main/tools/process-wrapper.c"],
    copts = ["-std=c99"],
)

cc_binary(
    name = "build-runfiles",
    srcs = ["main/tools/build-runfiles.cc"],
)

link_jni_cmd = "DIR=%s && cp $${DIR}/jni.h $(location jni.h) && cp $${DIR}/%s/jni_md.h $(location jni_md.h)"

genrule(
    name = "copy_link_jni_headers",
    srcs = [],
    outs = ["jni.h", "jni_md.h"],
    cmd = select({
        ":darwin": link_jni_cmd % ("/Library/Java/JavaVirtualMachines/jdk1.8.0_20.jdk/Contents/Home/include", "darwin"),
        "//conditions:default": link_jni_cmd % ("/usr/lib/jvm/java-7-openjdk-amd64/include", "linux"),
    }),
)

LIBARCHIVE_FILES = [
    "libarchive.a",
    "liblzo2.a",
    "liblzma.a",
    "libcharset.a",
    "libbz2.a",
    "libxml2.a",
    "libz.a",
    "libiconv.a",
]

genrule(
    name = "copy_static_libarchive",
    srcs = [],
    outs = LIBARCHIVE_FILES,
    cmd = select({
        ":darwin": "cp " + " ".join(["/opt/local/lib/" + x for x in LIBARCHIVE_FILES]) + " $(@D)",
        "//conditions:default": "touch $(OUTS)",
    }),
)

cc_binary(
    name = "libunix.so",
    srcs = select({
        ":darwin" : [
            "main/native/localsocket.cc",
            "main/native/process.cc",
            "main/native/unix_jni.cc",
            "main/native/unix_jni_darwin.cc",
            "jni.h",
            "jni_md.h",
        ],
        "//conditions:default" : [
            "main/native/localsocket.cc",
            "main/native/process.cc",
            "main/native/unix_jni.cc",
            "main/native/unix_jni_linux.cc",
            "jni.h",
            "jni_md.h",
        ],
    }),
    deps = [
        ":md5",
    ],
    copts = [
        "-fPIC",
        "-DBLAZE_JAVA_CPU=\"k8\"",
        "-DBLAZE_OPENSOURCE=1",
    ],
    includes = [
        ".", # for the jni headers
        "main/cpp",
    ],
    linkshared = 1,
)

cc_library(
    name = "util",
    hdrs = [
        "main/cpp/util/numbers.h",
        "main/cpp/util/port.h",
        "main/cpp/util/strings.h",
    ],
    srcs = [
        "main/cpp/util/numbers.cc",
        "main/cpp/util/port.cc",
        "main/cpp/util/strings.cc",
    ],
    includes = [
        "main/cpp",
    ],
    copts = [
        "-DBLAZE_OPENSOURCE=1",
    ],
)

cc_library(
    name = "md5",
    srcs = ["main/cpp/util/md5.cc"],
    hdrs = ["main/cpp/util/md5.h"],
    includes = ["main/cpp"],
)

cc_binary(
    name = "client",
    srcs = select({
        ":darwin": [
            "main/cpp/blaze.cc",
            "main/cpp/blaze_startup_options.cc",
            "main/cpp/blaze_startup_options_common.cc",
            "main/cpp/blaze_util.cc",
            "main/cpp/blaze_util_darwin.cc",
            "main/cpp/option_processor.cc",
            "main/cpp/util/file.cc",
            ":copy_static_libarchive"
        ],
        "//conditions:default": [
            "main/cpp/blaze.cc",
            "main/cpp/blaze_startup_options.cc",
            "main/cpp/blaze_startup_options_common.cc",
            "main/cpp/blaze_util.cc",
            "main/cpp/blaze_util_linux.cc",
            "main/cpp/option_processor.cc",
            "main/cpp/util/file.cc",
        ],
    }),
    deps = [
        ":util",
        ":md5",
    ],
    copts = [
        "-DBLAZE_JAVA_CPU=\\\"k8\\\"",
        "-DBLAZE_OPENSOURCE=1",
        "-I/opt/local/include"
    ],
    includes = [
        "main/cpp",
    ],
    linkopts = select({
        ":darwin" : [
        ],
        "//conditions:default" : [
            "-larchive",
            "-lrt",
        ],
    }),
)

# Packaging

genrule(
    name = "client-info-file",
    outs = ["client_info"],
    cmd = "touch $@",
    executable = 1,
)

# HACK for Mac: copy libunix.so to libunix.dylib. We'll need to come up with a
# way to support platform-specific dynamic library extensions.
genrule(
    name = "mac-compat",
    srcs = ["libunix.so"],
    outs = ["libunix.dylib"],
    cmd = "cp $< $@",
)

md5_cmd = "set -e -o pipefail && cat $(SRCS) | %s | awk '{ print $$1; }' > $@"

genrule(
    name = "install_base_key-file",
    srcs = [
        "client",
        "libblaze.jar",
        "libunix.so",
        "libunix.dylib",
        "build-runfiles",
        "process-wrapper",
        "client_info",
        "main/tools/build_interface_so",
    ],
    outs = ["install_base_key"],
    cmd = select({
        ":darwin": md5_cmd % "/sbin/md5",
        "//conditions:default": md5_cmd % "md5sum",
    })
)

genrule(
    name = "package-zip",
    srcs = [
        "libblaze.jar",
        "libunix.so",
        "libunix.dylib",
        "build-runfiles",
        "process-wrapper",
        "client_info",
        "main/tools/build_interface_so",
        "install_base_key",
    ],
    outs = ["package.zip"],
    cmd = "zip -qj $@ $(SRCS)",
)

genrule(
    name = "bazel-bin",
    srcs = [
        "client",
        "package-zip",
    ],
    outs = ["bazel"],
    cmd = "cat client package-zip > $@ && zip -qA $@",
    executable = 1,
)

config_setting(
    name = "darwin",
    values = { "cpu": "darwin" }
)

