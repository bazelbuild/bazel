load("tools/build_rules/genproto", "genproto")

load("tools/build_rules/java_rules_oss", "java_library", "java_binary", "java_test")

# The file assumes there is a directory (or symlink) src/javadir.
# Use this command to use local java:
#   ln -s $(which java | sed 's_/bin/java'__) src/javadir

JAVA_HOME = "src/javadir"

# Proto

[genproto(
    name = "proto_" + proto_file,
    src = "main/protobuf/" + proto_file + ".proto",
) for proto_file in [
    "build",
    "crosstool_config",
    "extra_actions_base",
    "test_status",
    "bundlemerge",
    "xcodegen",
]]

# Java

# TODO(bazel-team): Split this into multiple rules.
java_library(
    name = "xcode-common",
    srcs = glob([
        "tools/xcode-common/java/com/google/devtools/build/xcode/util/*.java",
        "tools/xcode-common/java/com/google/devtools/build/xcode/common/*.java",
    ]),
    jars = [
        "//third_party",
    ],
)

java_library(
    name = "bazel-core",
    srcs = glob([
        "main/**/*.java",
    ]),
    jars = [
        "//third_party",
    ],
    deps = [
        ":proto_build",
        ":proto_bundlemerge",
        ":proto_crosstool_config",
        ":proto_extra_actions_base",
        ":proto_test_status",
        ":proto_xcodegen",
        ":xcode-common",
    ],
)

java_binary(
    name = "bazel-main",
    main_class = "com.google.devtools.build.lib.bazel.BazelMain",
    deps = [
        ":bazel-core",
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

link_jni_cmd = ("cp tools/jdk/jdk/include/jni.h $(location jni.h) && " +
                "cp tools/jdk/jdk/include/%s/jni_md.h $(location jni_md.h)")

genrule(
    name = "copy_link_jni_headers",
    srcs = [
        "//tools/jdk:headers",
    ],
    outs = [
        "jni.h",
        "jni_md.h",
    ],
    cmd = select({
        ":darwin": link_jni_cmd % "darwin",
        "//conditions:default": link_jni_cmd % "linux",
    }),
)

cc_binary(
    name = "libunix.so",
    srcs = select({
        ":darwin": [
            "main/native/localsocket.cc",
            "main/native/process.cc",
            "main/native/unix_jni.cc",
            "main/native/unix_jni_darwin.cc",
            "jni.h",
            "jni_md.h",
        ],
        "//conditions:default": [
            "main/native/localsocket.cc",
            "main/native/process.cc",
            "main/native/unix_jni.cc",
            "main/native/unix_jni_linux.cc",
            "jni.h",
            "jni_md.h",
        ],
    }),
    copts = [
        "-fPIC",
        "-DBLAZE_JAVA_CPU=\"k8\"",
        "-DBLAZE_OPENSOURCE=1",
    ],
    includes = [
        ".",  # for the jni headers
        "main/cpp",
    ],
    linkshared = 1,
    deps = [
        ":md5",
    ],
)

cc_library(
    name = "util",
    srcs = [
        "main/cpp/util/numbers.cc",
        "main/cpp/util/port.cc",
        "main/cpp/util/strings.cc",
    ],
    hdrs = [
        "main/cpp/util/numbers.h",
        "main/cpp/util/port.h",
        "main/cpp/util/strings.h",
    ],
    copts = [
        "-DBLAZE_OPENSOURCE=1",
    ],
    includes = [
        "main/cpp",
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
    copts = [
        "-DBLAZE_JAVA_CPU=\\\"k8\\\"",
        "-DBLAZE_OPENSOURCE=1",
    ],
    includes = [
        "main/cpp",
    ],
    linkopts = select({
        ":darwin": [
        ],
        "//conditions:default": [
            "-larchive",
            "-lrt",
        ],
    }),
    deps = select({
        ":darwin": [
            ":md5",
            ":util",
            "//fromhost:libarchive",
        ],
        "//conditions:default": [
            ":md5",
            ":util",
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

# TODO: The jar file name in the client needs to match this. Can we automate that?
genrule(
    name = "rename-bazel-jar",
    srcs = ["bazel-main_deploy.jar"],
    outs = ["libblaze.jar"],
    cmd = "cp $< $@",
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
    }),
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
    values = {"cpu": "darwin"},
)

# Tests

java_library(
    name = "testutil",
    srcs = glob([
        "test/java/com/google/devtools/build/lib/testutil/*.java",
    ]),
    jars = [
        ":bazel-core",
        "//third_party",
        "//third_party:third_party_test_utils",
    ],
)

java_test(
    name = "skyframe_test",
    srcs = glob([
        "test/java/com/google/devtools/build/skyframe/*.java",
    ]),
    args = ["com.google.devtools.build.skyframe.AllTests"],
    jars = [
        "//third_party",
        "//third_party:third_party_test_utils",
    ],
    deps = [
        ":bazel-core",
        ":testutil",
    ],
)

java_test(
    name = "options_test",
    srcs = glob([
        "test/java/com/google/devtools/common/options/*.java",
    ]),
    args = ["com.google.devtools.common.options.AllTests"],
    jars = [
        "//third_party",
        "//third_party:third_party_test_utils",
    ],
    deps = [
        ":bazel-core",
        ":testutil",
    ],
)

test_prefix = "test/java/com/google/devtools/build/lib"

java_test(
    name = "foundations_test",
    srcs = glob(
        ["%s/%s" % (test_prefix, p) for p in [
            "concurrent/*.java",
            "collect/*.java",
            "collect/nestedset/*.java",
            "events/*.java",
            "testutiltests/*.java",
            "unix/*.java",
            "util/*.java",
            "util/io/*.java",
            "vfs/*.java",
            "vfs/util/*.java",
            "vfs/inmemoryfs/*.java",
            "AllTests.java",
        ]],
        # java_rules_oss doesn't support resource loading with
        # qualified paths.
        exclude = [
            test_prefix + f
            for f in [
                "/util/DependencySetWindowsTest.java",
                "/util/ResourceFileLoaderTest.java",
                "/vfs/PathFragmentWindowsTest.java",
                "/vfs/PathWindowsTest.java",
            ]
        ],
    ),
    args = ["com.google.devtools.build.lib.AllTests"],
    data = glob([
        test_prefix + "/vfs/*.zip",
    ]),
    jars = [
        "//third_party",
        "//third_party:third_party_test_utils",
    ],
    resources = [
        ":libunix.dylib",
        ":libunix.so",
    ],
    deps = [
        ":bazel-core",
        ":testutil",
    ],
)

java_test(
    name = "windows_test",
    srcs = glob([test_prefix + "/vfs/util/*.java"]) +
           ["%s/%s" % (test_prefix, p) for p in [
               "util/DependencySetWindowsTest.java",
               "vfs/PathFragmentWindowsTest.java",
               "vfs/PathWindowsTest.java",
               "AllTests.java",
           ]],
    args = [
        "com.google.devtools.build.lib.AllTests",
    ],
    jars = [
        "//third_party",
        "//third_party:third_party_test_utils",
    ],
    jvm_flags = ["-Dblaze.os=Windows"],
    resources = [
        ":libunix.dylib",
        ":libunix.so",
    ],
    deps = [
        ":bazel-core",
        ":testutil",
    ],
)
