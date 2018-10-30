package(default_visibility = ["//visibility:public"])

load(
    "//tools/jdk:default_java_toolchain.bzl",
    "default_java_toolchain",
    "java_runtime_files",
    "JDK8_JVM_OPTS",
    "JDK9_JVM_OPTS",
    "DEFAULT_JAVACOPTS",
    "bootclasspath",
)

# Used to distinguish toolchains used for Java development, ie the JavaToolchainProvider.
toolchain_type(name = "toolchain_type")

# Used to distinguish toolchains used for Java execution, ie the JavaRuntimeInfo.
toolchain_type(name = "runtime_toolchain_type")

java_runtime_alias(name = "current_java_runtime")

java_host_runtime_alias(name = "current_host_java_runtime")

java_toolchain_alias(name = "current_java_toolchain")

# Used to set --host_javabase or --javabase to a local JDK without having to define
# a custom java_runtime rule.
# E.g.:
# bazel build --host_javabase=@bazel_tools//tools/jdk:absolute_javabase \
#   --define=ABSOLUTE_JAVABASE=<path to JDK> ...
java_runtime(
    name = "absolute_javabase",
    java_home = "$(ABSOLUTE_JAVABASE)",
)

filegroup(
    name = "BUILD-jdk",
    srcs = [":BUILD"],
)

# This is necessary to get the *host* Java runtime. Depending on
# //tools/jdk:current_java_runtime from an attribute with the host transition
# does not work because the dependency is determined based on the configuration
# *before* the transition.
alias(
    name = "java_runtime_alias",
    actual = "@bazel_tools//tools/jdk:current_java_runtime",
)

java_runtime_files(
    name = "jni_header",
    srcs = ["include/jni.h"],
)

java_runtime_files(
    name = "jni_md_header-darwin",
    srcs = ["include/darwin/jni_md.h"],
)

java_runtime_files(
    name = "jni_md_header-linux",
    srcs = ["include/linux/jni_md.h"],
)

java_runtime_files(
    name = "jni_md_header-windows",
    srcs = ["include/win32/jni_md.h"],
)

java_runtime_files(
    name = "jni_md_header-freebsd",
    srcs = ["include/freebsd/jni_md.h"],
)

alias(
    name = "java",
    actual = "@local_jdk//:java",
)

alias(
    name = "jar",
    actual = "@local_jdk//:jar",
)

alias(
    name = "javac",
    actual = "@local_jdk//:javac",
)

# On Windows, executables end in ".exe", but the label we reach it through
# must be platform-independent. Thus, we create a little filegroup that
# contains the appropriate platform-dependent file.
filegroup(
    name = "ijar",
    srcs = select({
        "//src/conditions:remote": ["//third_party/ijar:ijar"],
        "//conditions:default": glob(["ijar/*"]),
    }),
)

# On Windows, Java implementation of singlejar is used. We create a little
# filegroup that contains the appropriate platform-dependent file.
# Once https://github.com/bazelbuild/bazel/issues/2241 is fixed (that is,
# the native singlejar is used on windows), this file group can be reused since
# on Windows, executables end in ".exe", but the label we reach it through
# must be platform-independent.
filegroup(
    name = "singlejar",
    srcs = select({
        "//src/conditions:remote": [":singlejar_remote"],
        "//conditions:default": glob(["singlejar/*"]),
    }),
)

# This is a separate filegroup from :singlejar above since Bazel doesn't
# support nested selects (we need a logical "remote AND windows" condition).
filegroup(
    name = "singlejar_remote",
    srcs = select({
        # TODO(jsharpe): Switch to //src/tools/singlejar:singlejar once
        # native singlejar works on Windows.
        "//src/conditions:windows": glob(["singlejar/*"]),
        "//conditions:default": ["//src/tools/singlejar:singlejar"],
    }),
)

filegroup(
    name = "genclass",
    srcs = ["//tools/jdk:GenClass_deploy.jar"],
)

filegroup(
    name = "turbine",
    srcs = ["//tools/jdk:turbine_deploy.jar"],
)

filegroup(
    name = "turbine_direct",
    srcs = glob(["turbine_direct*"]),
)

filegroup(
    name = "javabuilder",
    srcs = ["//tools/jdk:JavaBuilder_deploy.jar"],
)

filegroup(
    name = "vanillajavabuilder",
    srcs = ["//tools/jdk:VanillaJavaBuilder_deploy.jar"],
)

BOOTCLASS_JARS = [
    "rt.jar",
    "resources.jar",
    "jsse.jar",
    "jce.jar",
    "charsets.jar",
]

# TODO(cushon): this isn't compatible with JDK 9
alias(
    name = "bootclasspath",
    actual = "@local_jdk//:bootclasspath",
)

alias(
    name = "extclasspath",
    actual = "@local_jdk//:extdir",
)

# TODO(cushon): migrate to extclasspath and delete
alias(
    name = "extdir",
    actual = "@local_jdk//:extdir",
)

filegroup(
    name = "langtools",
    srcs = ["//third_party/java/jdk/langtools:javac_jar"],
)

java_import(
    name = "langtools-neverlink",
    jars = [":langtools"],
    neverlink = 1,
)

alias(
    name = "jre",
    actual = "@local_jdk//:jre",
)

alias(
    name = "jdk",
    actual = "@local_jdk//:jdk",
)

alias(
    name = "host_jdk",
    actual = "@embedded_jdk//:jdk",
)

bootclasspath(
    name = "platformclasspath",
    src = "DumpPlatformClassPath.java",
    host_javabase = "current_java_runtime",
    target_javabase = "current_java_runtime",
)

default_java_toolchain(
    name = "toolchain_hostjdk8",
    bootclasspath = [":platformclasspath"],
    jvm_opts = JDK8_JVM_OPTS,
    source_version = "8",
    target_version = "8",
)

# Default to the Java 8 language level.
# TODO(cushon): consider if/when we should increment this?
default_java_toolchain(
    name = "toolchain_hostjdk9",
    bootclasspath = [":platformclasspath"],
    jvm_opts = JDK9_JVM_OPTS,
    source_version = "8",
    target_version = "8",
)

# The 'vanilla' toolchain is an unsupported alternative to the default.
#
# It does not provider any of the following features:
#   * Error Prone
#   * Strict Java Deps
#   * Header Compilation
#   * Reduced Classpath Optimization
#
# It uses the version of javac from the `--host_javabase` instead of the
# embedded javac, which may not be source- or bug-compatible with the embedded
# javac.
#
# However it does allow using a wider range of `--host_javabase`s, including
# versions newer than the current embedded JDK.
default_java_toolchain(
    name = "toolchain_vanilla",
    forcibly_disable_header_compilation = True,
    javabuilder = [":vanillajavabuilder"],
    jvm_opts = [],
)

RELEASES = (8, 9)

[
    default_java_toolchain(
        name = "toolchain_java%d" % release,
        bootclasspath = [":platformclasspath"],
        jvm_opts = JDK9_JVM_OPTS,
        source_version = "%s" % release,
        target_version = "%s" % release,
    )
    for release in RELEASES
]

alias(
    name = "toolchain",
    actual = "//external:java_toolchain",
)

filegroup(
    name = "srcs",
    srcs = [
        "BUILD-jdk",  # Tools are build from the workspace for tests.
        "DumpPlatformClassPath.java",
        "default_java_toolchain.bzl",
        "proguard_whitelister.py",
        "proguard_whitelister_test.py",
        "proguard_whitelister_test_input.cfg",
    ],
)

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
)

py_binary(
    name = "proguard_whitelister",
    srcs = [
        "proguard_whitelister.py",
    ],
    deps = [
        "//third_party/py/gflags",
    ],
)

py_test(
    name = "proguard_whitelister_test",
    srcs = ["proguard_whitelister_test.py"],
    data = ["proguard_whitelister_test_input.cfg"],
    deps = [
        ":proguard_whitelister",
    ],
)

# For java coverage
alias(
    name = "jacoco-blaze-agent",
    actual = "//third_party/java/jacoco:blaze-agent",
)

java_import(
    name = "JacocoCoverage",
    jars = [":JacocoCoverage_deploy.jar"],
)

test_suite(
    name = "windows_tests",
    tags = [
        "-no_windows",
        "-slow",
    ],
    visibility = ["//visibility:private"],
)

test_suite(
    name = "all_windows_tests",
    tests = [":windows_tests"],
    visibility = ["//tools:__pkg__"],
)

# Replacement for the embedded JDK as host javabase.
alias(
    name = "remote_jdk",
    actual = select(
        {
            "//src/conditions:darwin": "@remotejdk_macos//:jdk",
            "//src/conditions:darwin_x86_64": "@remotejdk_macos//:jdk",
            "//src/conditions:windows": "@remotejdk_win//:jdk",
            "//src/conditions:linux_aarch64": "@remotejdk_linux_aarch64//:jdk",
            "//src/conditions:linux_x86_64": "@remotejdk_linux//:jdk",
        },
        no_match_error = "Could not find a JDK for host execution environment, please explicitly" +
                         " provide one using `--host_javabase.`",
    ),
    visibility = ["//visibility:public"],
)

# A JDK 10 for use as a --host_javabase.
alias(
    name = "remote_jdk10",
    actual = select(
        {
            "//src/conditions:darwin": "@remotejdk10_macos//:jdk",
            "//src/conditions:darwin_x86_64": "@remotejdk10_macos//:jdk",
            "//src/conditions:windows": "@remotejdk10_win//:jdk",
            "//src/conditions:linux_aarch64": "@remotejdk10_linux_aarch64//:jdk",
            "//src/conditions:linux_x86_64": "@remotejdk10_linux//:jdk",
        },
        no_match_error = "Could not find a JDK for host execution environment, please explicitly" +
                         " provide one using `--host_javabase.`",
    ),
    visibility = ["//visibility:public"],
)
