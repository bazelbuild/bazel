# This is a quick and dirty way to compile Bazel using only genrule.
# We will need proper proto and java rules.

load('rules/java_library_oss', 'java_library')

# The file assumes there is a directory (or symlink) src/javadir.
# Use this command to use local java:
#   ln -s $(which java | sed 's_/bin/java'__) src/javadir

JAVA_HOME = "src/javadir"

platform = "linux"

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
protoc -Isrc/main/protobuf/ --java_out=. %s

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
    cmd = proto_cmd % "$(location main/protobuf/build.proto)",
)

genrule(
    name = "proto_crosstool_config",
    srcs = [
        "main/protobuf/crosstool_config.proto",
        "//third_party:protobuf",
    ],
    outs = ["libCrosstoolConfig.jar"],
    cmd = proto_cmd % "$(location main/protobuf/crosstool_config.proto)",
)

genrule(
    name = "proto_extra_actions_base",
    srcs = [
        "main/protobuf/extra_actions_base.proto",
        "//third_party:protobuf",
    ],
    outs = ["libExtraActionsBase.jar"],
    cmd = proto_cmd % "$(location main/protobuf/extra_actions_base.proto)",
)

genrule(
    name = "proto_test_status",
    srcs = [
        "main/protobuf/test_status.proto",
        "//third_party:protobuf",
    ],
    outs = ["libTestStatus.jar"],
    cmd = proto_cmd % "$(location main/protobuf/test_status.proto)",
)

genrule(
    name = "proto_bundlemerge",
    srcs = [
        "main/protobuf/bundlemerge.proto",
        "//third_party:protobuf",
    ],
    outs = ["libBundleMerge.jar"],
    cmd = proto_cmd % "$(location main/protobuf/bundlemerge.proto)",
)

genrule(
    name = "proto_xcodegen",
    srcs = [
        "main/protobuf/xcodegen.proto",
        "//third_party:protobuf",
    ],
    outs = ["libXCodeGen.jar"],
    cmd = proto_cmd % "$(location main/protobuf/xcodegen.proto)",
)

# Java

# TODO(bazel-team): Split this into multiple rules.
java_library(
    name = "blaze",
    java_home = JAVA_HOME,
    srcdirs = [
        "src/main/java",
        "src/tools/singlejar/java",
        "src/tools/xcode-common/java",
        "src/tools/xcodegen/java",
        "src/third_party/buck/java",
        "src/third_party/dd_plist/java",
    ],
    main_class = "com.google.devtools.build.lib.bazel.BazelMain",
    srcs = glob([
        "main/**/*.java",
        "tools/singlejar/java/**/*.java",
        "tools/xcode-common/java/**/*.java",
        "tools/xcodegen/java/**/*.java",
        "third_party/buck/**/*.java",
        "third_party/dd_plist/java/**/*.java",
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

cc_binary(
    name = "libunix.so",
    srcs = [
        "main/cpp/util/md5.cc",
        "main/native/localsocket.cc",
        "main/native/process.cc",
        "main/native/unix_jni.cc",
        "main/native/unix_jni_%s.cc" % platform,
    ],
    copts = [
        "-fPIC",
        "-DBLAZE_JAVA_CPU=\"k8\"",
        "-DBLAZE_OPENSOURCE=1",
    ],
    includes = [
        "javadir/include",
        "javadir/include/%s" % platform,
        "main/cpp",
    ],
    linkopts = ["-lstdc++"],
    linkshared = 1,
)

cc_binary(
    name = "client",
    srcs = [
        "main/cpp/blaze.cc",
        "main/cpp/blaze_startup_options.cc",
        "main/cpp/blaze_startup_options_common.cc",
        "main/cpp/blaze_util.cc",
        "main/cpp/blaze_util_%s.cc" % platform,
        "main/cpp/option_processor.cc",
        "main/cpp/util/file.cc",
        "main/cpp/util/md5.cc",
        "main/cpp/util/numbers.cc",
        "main/cpp/util/port.cc",
        "main/cpp/util/strings.cc",
    ],
    copts = [
        "-DBLAZE_JAVA_CPU=\\\"k8\\\"",
        "-DBLAZE_OPENSOURCE=1",
    ],
    includes = [
        "main/cpp",
    ],
    linkopts = [
        "-larchive",
        "-lstdc++",
        "-lrt",
    ],
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
    cmd = "cat $(SRCS) | md5sum | awk '{ print $$1; }' > $@",
)

genrule(
    name = "package-zip",
    srcs = [
        "libblaze.jar",
        "libunix.so",
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
