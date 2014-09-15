# This is a quick and dirty way to compile Bazel using only genrule.
# We will need proper proto and java rules.

load('rules/java_library_oss', 'java_library')

# The file assumes there is a directory (or symlink) src/javadir.
# Use this command to use local java:
#   ln -s $(which java | sed 's_/bin/java'__) src/javadir

JAVA_HOME = "src/javadir"

protoc = "protoc"
protoc_darwin = "/opt/local/bin/protoc"

# Common code that filters the .jar files from the srcs.
# Once we move have proper rules, we can use a separate attribute (e.g. deps).
header = """
set -e
JARS=""
for i in $(SRCS); do
  if echo "$$i" | grep -q '\.jar'; then
    JARS="$${JARS}:$$i"
  fi
done

# TODO(bazel-team): Remove hard-coded path to Java.
JAVA=/usr/bin/javac
mkdir -p build_output
"""

# Proto

# Run protoc on the input (%s expanded), compile the generated source files
# and package them in a .jar file.
proto_cmd = header + """
%s -Isrc/main/protobuf/ --java_out=. %s

JAVA_FILES=$$(find com/ -name '*.java')
$${JAVA} -classpath $${JARS} $${JAVA_FILES} -d build_output
jar cf $@ -C build_output com
"""

genrule(
    name = "proto_build",
    srcs = [
        "main/protobuf/build.proto",
        "//third_party:protobuf",
    ],
    outs = ["libBuild.jar"],
    cmd = select({
        ":darwin": proto_cmd % (protoc_darwin, "$(location main/protobuf/build.proto)"),
        "//conditions:default": proto_cmd % (protoc, "$(location main/protobuf/build.proto)"),
    }),
)

genrule(
    name = "proto_crosstool_config",
    srcs = [
        "main/protobuf/crosstool_config.proto",
        "//third_party:protobuf",
    ],
    outs = ["libCrosstoolConfig.jar"],
    cmd = select({
        ":darwin": proto_cmd % (protoc_darwin, "$(location main/protobuf/crosstool_config.proto)"),
        "//conditions:default": proto_cmd % (protoc, "$(location main/protobuf/crosstool_config.proto)"),
    }),
)

genrule(
    name = "proto_extra_actions_base",
    srcs = [
        "main/protobuf/extra_actions_base.proto",
        "//third_party:protobuf",
    ],
    outs = ["libExtraActionsBase.jar"],
    cmd = select({
        ":darwin": proto_cmd % (protoc_darwin, "$(location main/protobuf/extra_actions_base.proto)"),
        "//conditions:default": proto_cmd % (protoc, "$(location main/protobuf/extra_actions_base.proto)"),
    }),
)

genrule(
    name = "proto_test_status",
    srcs = [
        "main/protobuf/test_status.proto",
        "//third_party:protobuf",
    ],
    outs = ["libTestStatus.jar"],
    cmd = select({
        ":darwin": proto_cmd % (protoc_darwin, "$(location main/protobuf/test_status.proto)"),
        "//conditions:default": proto_cmd % (protoc, "$(location main/protobuf/test_status.proto)"),
    }),
)

genrule(
    name = "proto_bundlemerge",
    srcs = [
        "main/protobuf/bundlemerge.proto",
        "//third_party:protobuf",
    ],
    outs = ["libBundleMerge.jar"],
    cmd = select({
        ":darwin": proto_cmd % (protoc_darwin, "$(location main/protobuf/bundlemerge.proto)"),
        "//conditions:default": proto_cmd % (protoc, "$(location main/protobuf/bundlemerge.proto)"),
    }),
)

genrule(
    name = "proto_xcodegen",
    srcs = [
        "main/protobuf/xcodegen.proto",
        "//third_party:protobuf",
    ],
    outs = ["libXCodeGen.jar"],
    cmd = select({
        ":darwin": proto_cmd % (protoc_darwin, "$(location main/protobuf/xcodegen.proto)"),
        "//conditions:default": proto_cmd % (protoc, "$(location main/protobuf/xcodegen.proto)"),
    }),
)

# Java

# TODO(bazel-team): Split this into multiple rules.
java_library(
    name = "blaze",
    java_home = JAVA_HOME,
    main_class = "com.google.devtools.build.lib.bazel.BazelMain",
    srcs = glob([
        "main/**/*.java",
        "tools/xcode-common/java/com/google/devtools/build/xcode/util/*.java",
        "tools/xcode-common/java/com/google/devtools/build/xcode/common/*.java",
    ]),
    deps = [
        "//third_party",
        ":proto_build",
        ":proto_bundlemerge",
        ":proto_crosstool_config",
        ":proto_extra_actions_base",
        ":proto_test_status",
        ":proto_xcodegen",
    ],
)

# C++

cc_binary(
    name = "cc-process-wrapper",
    srcs = ["main/tools/process-wrapper.c"],
    copts = ["-std=c99"],
)

cc_binary(
    name = "build-runfiles",
    srcs = ["main/tools/build-runfiles.cc"],
    linkopts = ["-lstdc++"],
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
        "//conditions:default": "touch $OUTS",
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
    linkopts = ["-lstdc++"],
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
            "-lstdc++",
        ],
        "//conditions:default" : [
            "-larchive",
            "-lstdc++",
            "-lrt",
        ],
    }),
)

# Packaging

genrule(
    name = "alarm-file",
    outs = ["alarm"],
    cmd = "touch $@",
    executable = 1,
)

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
        "cc-process-wrapper",
        "alarm",
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
        "cc-process-wrapper",
        "alarm",
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

