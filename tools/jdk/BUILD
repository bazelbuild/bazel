load("@rules_java//java:defs.bzl", "java_import", "java_runtime", "java_toolchain")
load(
    "//tools/jdk:default_java_toolchain.bzl",
    "JDK8_JVM_OPTS",
    "bootclasspath",
    "default_java_toolchain",
    "java_runtime_files",
)
load(
    "//tools/jdk:java_toolchain_alias.bzl",
    "java_host_runtime_alias",
    "java_runtime_alias",
    "java_toolchain_alias",
    "legacy_java_runtime_alias",
    "legacy_java_toolchain_alias",
)
load(
    "//tools/jdk:remote_java_tools_aliases.bzl",
    "remote_java_tools_filegroup",
    "remote_java_tools_java_import",
)
load("@rules_cc//cc:defs.bzl", "cc_binary")
load("//tools/python:private/defs.bzl", "py_binary", "py_test")

package(default_visibility = ["//visibility:public"])

# Used to distinguish toolchains used for Java development, ie the JavaToolchainProvider.
toolchain_type(name = "toolchain_type")

# Used to distinguish toolchains used for Java execution, ie the JavaRuntimeInfo.
toolchain_type(name = "runtime_toolchain_type")

toolchain(
    name = "dummy_java_toolchain",
    toolchain = ":dummy_toolchain",
    toolchain_type = ":toolchain_type",
)

cc_binary(name = "dummy_binary")

java_toolchain(
    name = "dummy_toolchain",
    genclass = [":dummy_binary"],
    ijar = [":dummy_binary"],
    javabuilder = [":dummy_binary"],
    javac = [":dummy_binary"],
    singlejar = [":dummy_binary"],
)

toolchain(
    name = "dummy_java_runtime_toolchain",
    toolchain = ":dummy_java_runtime",
    toolchain_type = ":runtime_toolchain_type",
)

java_runtime(name = "dummy_java_runtime")

java_runtime_alias(name = "current_java_runtime")

java_host_runtime_alias(name = "current_host_java_runtime")

java_toolchain_alias(name = "current_java_toolchain")

# This exists to support the migration to toolchain resolution.
# TODO(cushon): delete once the migration is complete.
legacy_java_runtime_alias(name = "legacy_current_java_runtime")

legacy_java_toolchain_alias(name = "legacy_current_java_toolchain")

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

java_runtime_files(
    name = "jni_md_header-openbsd",
    srcs = ["include/openbsd/jni_md.h"],
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

alias(
    name = "javadoc",
    actual = "@local_jdk//:javadoc",
)

# On Windows, executables end in ".exe", but the label we reach it through
# must be platform-independent. Thus, we create a little filegroup that
# contains the appropriate platform-dependent file.
remote_java_tools_filegroup(
    name = "ijar",
    target = ":ijar",
)

# On Windows, Java implementation of singlejar is used. We create a little
# filegroup that contains the appropriate platform-dependent file.
# Once https://github.com/bazelbuild/bazel/issues/2241 is fixed (that is,
# the native singlejar is used on windows), this file group can be reused since
# on Windows, executables end in ".exe", but the label we reach it through
# must be platform-independent.
remote_java_tools_filegroup(
    name = "singlejar",
    target = ":singlejar",
)

exports_files(["BUILD.java_tools"])

remote_java_tools_filegroup(
    name = "genclass",
    target = ":GenClass",
)

remote_java_tools_filegroup(
    name = "GenClass_deploy.jar",
    target = ":java_tools/GenClass_deploy.jar",
)

remote_java_tools_filegroup(
    name = "bazel-singlejar_deploy.jar",
    target = ":java_tools/bazel-singlejar_deploy.jar",
)

remote_java_tools_filegroup(
    name = "turbine",
    target = ":Turbine",
)

remote_java_tools_filegroup(
    name = "turbine_deploy.jar",
    target = ":java_tools/turbine_deploy.jar",
)

remote_java_tools_filegroup(
    name = "turbine_direct",
    target = ":TurbineDirect",
)

remote_java_tools_filegroup(
    name = "turbine_direct_binary_deploy.jar",
    target = ":java_tools/turbine_direct_binary_deploy.jar",
)

remote_java_tools_filegroup(
    name = "javabuilder",
    target = ":JavaBuilder",
)

remote_java_tools_filegroup(
    name = "JavaBuilder_deploy.jar",
    target = ":java_tools/JavaBuilder_deploy.jar",
)

remote_java_tools_filegroup(
    name = "vanillajavabuilder",
    target = ":VanillaJavaBuilder",
)

remote_java_tools_filegroup(
    name = "javac_jar",
    target = ":javac_jar",
)

remote_java_tools_filegroup(
    name = "jdk_compiler_jar",
    target = ":jdk_compiler_jar",
)

remote_java_tools_filegroup(
    name = "java_compiler_jar",
    target = ":java_compiler_jar",
)

remote_java_tools_java_import(
    name = "JacocoCoverageRunner",
    target = ":java_tools/JacocoCoverage_jarjar_deploy.jar",
)

remote_java_tools_java_import(
    name = "JacocoCoverage",
    target = ":java_tools/JacocoCoverage_jarjar_deploy.jar",
)

remote_java_tools_java_import(
    name = "TestRunner",
    target = ":java_tools/Runner_deploy.jar",
)

remote_java_tools_filegroup(
    name = "TestRunner_deploy.jar",
    target = ":java_tools/Runner_deploy.jar",
)

alias(
    name = "proguard",
    actual = select({
        "//src/conditions:darwin": "@remote_java_tools_darwin//:proguard",
        "//src/conditions:darwin_x86_64": "@remote_java_tools_darwin//:proguard",
        "//src/conditions:windows": "@remote_java_tools_windows//:proguard",
        "//src/conditions:linux_x86_64": "@remote_java_tools_linux//:proguard",
        # On different platforms the linux repository can be used because the
        # proguard deploy jar is platform-agnostic.
        "//conditions:default": "@remote_java_tools_linux//:proguard",
    }),
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
    actual = ":remote_jdk",
)

bootclasspath(
    name = "platformclasspath",
    src = "DumpPlatformClassPath.java",
    host_javabase = "current_java_runtime",
    target_javabase = "current_java_runtime",
)

default_java_toolchain(
    name = "toolchain_hostjdk8",
    jvm_opts = JDK8_JVM_OPTS,
    source_version = "8",
    target_version = "8",
)

# Default to the Java 8 language level.
# TODO(cushon): consider if/when we should increment this?
default_java_toolchain(
    name = "legacy_toolchain",
    source_version = "8",
    target_version = "8",
)

alias(
    name = "toolchain",
    actual = select({
        "//src/conditions:darwin": "@remote_java_tools_darwin//:toolchain",
        "//src/conditions:darwin_x86_64": "@remote_java_tools_darwin//:toolchain",
        "//src/conditions:windows": "@remote_java_tools_windows//:toolchain",
        "//src/conditions:linux_x86_64": "@remote_java_tools_linux//:toolchain",
        "//conditions:default": "@bazel_tools//tools/jdk:legacy_toolchain",
    }),
)

alias(
    name = "remote_toolchain",
    actual = select({
        "//src/conditions:darwin": "@remote_java_tools_darwin//:toolchain",
        "//src/conditions:darwin_x86_64": "@remote_java_tools_darwin//:toolchain",
        "//src/conditions:windows": "@remote_java_tools_windows//:toolchain",
        "//src/conditions:linux_x86_64": "@remote_java_tools_linux//:toolchain",
        "//conditions:default": "@bazel_tools//tools/jdk:legacy_toolchain",
    }),
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
    source_version = "",
    target_version = "",
)

RELEASES = (8, 9, 10, 11)

[
    default_java_toolchain(
        name = "toolchain_java%d" % release,
        source_version = "%s" % release,
        target_version = "%s" % release,
    )
    for release in RELEASES
]

filegroup(
    name = "srcs",
    srcs = [
        "BUILD.java_langtools",
        "BUILD.java_tools",
        "BUILD-jdk",  # Tools are build from the workspace for tests.
        "DumpPlatformClassPath.java",
        "default_java_toolchain.bzl",
        "java_toolchain_alias.bzl",
        "proguard_whitelister.py",
        "proguard_whitelister_test.py",
        "proguard_whitelister_test_input.pgcfg",
        "remote_java_tools_aliases.bzl",
        "toolchain_utils.bzl",
    ],
)

filegroup(
    name = "bzl_srcs",
    srcs = glob(["*.bzl"]),
    visibility = ["//tools:__pkg__"],
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
        "//third_party/py/abseil",
    ],
)

py_test(
    name = "proguard_whitelister_test",
    srcs = ["proguard_whitelister_test.py"],
    data = ["proguard_whitelister_test_input.pgcfg"],
    deps = [
        ":proguard_whitelister",
    ],
)

remote_java_tools_java_import(
    name = "JarJar",
    target = ":JarJar",
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

# A JDK 11 for use as a --host_javabase.
alias(
    name = "remote_jdk11",
    actual = select(
        {
            "//src/conditions:darwin": "@remotejdk11_macos//:jdk",
            "//src/conditions:darwin_x86_64": "@remotejdk11_macos//:jdk",
            "//src/conditions:windows": "@remotejdk11_win//:jdk",
            "//src/conditions:linux_aarch64": "@remotejdk11_linux_aarch64//:jdk",
            "//src/conditions:linux_x86_64": "@remotejdk11_linux//:jdk",
        },
        no_match_error = "Could not find a JDK for host execution environment, please explicitly" +
                         " provide one using `--host_javabase.`",
    ),
    visibility = ["//visibility:public"],
)
