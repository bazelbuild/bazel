load("@rules_java//java:defs.bzl", "java_import", "java_runtime", "java_toolchain")
load(
    "//tools/jdk:default_java_toolchain.bzl",
    "JDK8_JVM_OPTS",
    "JVM8_TOOLCHAIN_CONFIGURATION",
    "DEFAULT_TOOLCHAIN_CONFIGURATION",
    "VANILLA_TOOLCHAIN_CONFIGURATION",
    "PREBUILT_TOOLCHAIN_CONFIGURATION",
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
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")
load("//tools/python:private/defs.bzl", "py_binary", "py_test")

package(default_visibility = ["//visibility:public"])

# Used to distinguish toolchains used for Java development, ie the JavaToolchainProvider.
toolchain_type(name = "toolchain_type")

# Used to distinguish toolchains used for Java execution, ie the JavaRuntimeInfo.
toolchain_type(name = "runtime_toolchain_type")

# Points to either :runtime_toolchain_type or :legacy_current_java_runtime
java_runtime_alias(name = "current_java_runtime")

# Points to :current_java_runtime with host transition
java_host_runtime_alias(name = "current_host_java_runtime")

# Points to either :toolchain_type or :legacy_current_java_toolchain
java_toolchain_alias(name = "current_java_toolchain")

# This exists to support the migration to toolchain resolution.
# TODO(cushon): delete once the migration is complete.
# Points to --javabase/--host_javabase, defaults ":jdk"/":remote_jdk11"
legacy_java_runtime_alias(name = "legacy_current_java_runtime")

# Points to --java_toolchain/--host_java_toolchain, defaults ":toolchain"/":remote_toolchain"
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

# These individual jni_* targets are exposed for legacy reasons.
# Most users should depend on :jni.

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

# The Java native interface. Depend on this package if you #include <jni.h>.
#
# See test_jni in third_party/bazel/src/test/shell/bazel/bazel_java_test.sh for
# an example of using Bazel to build a Java program that calls a C function.
#
# TODO(ilist): use //src:condition:linux when released in Bazel
# Required by: "//src/main/native..."
cc_library(
    name = "jni",
    hdrs = [":jni_header"] + select({
        "//src/conditions:linux_aarch64": [":jni_md_header-linux"],
        "//src/conditions:linux_ppc64le": [":jni_md_header-linux"],
        "//src/conditions:linux_s390x": [":jni_md_header-linux"],
        "//src/conditions:linux_x86_64": [":jni_md_header-linux"],
        "//src/conditions:darwin": [":jni_md_header-darwin"],
        "//src/conditions:freebsd": [":jni_md_header-freebsd"],
        "//src/conditions:openbsd": [":jni_md_header-openbsd"],
        "//src/conditions:windows": [":jni_md_header-windows"],
        "//conditions:default": [],
    }),
    includes = ["include"] + select({
        "//src/conditions:linux_aarch64": ["include/linux"],
        "//src/conditions:linux_ppc64le": ["include/linux"],
        "//src/conditions:linux_s390x": ["include/linux"],
        "//src/conditions:linux_x86_64": ["include/linux"],
        "//src/conditions:darwin": ["include/darwin"],
        "//src/conditions:freebsd": ["include/freebsd"],
        "//src/conditions:openbsd": ["include/openbsd"],
        "//src/conditions:windows": ["include/win32"],
        "//conditions:default": [],
    }),
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

config_setting(
    name = "remote",
    values = {"define": "EXECUTOR=remote"},
)

# On Windows, executables end in ".exe", but the label we reach it through
# must be platform-independent. Thus, we create a little filegroup that
# contains the appropriate platform-dependent file.
alias(
   name = "ijar",
    actual = select({
        ":remote": "@remote_java_tools//:ijar_cc_binary",
        "//conditions:default": ":ijar_prebuilt_binary_or_cc_binary",
    }),
)

alias(
    name = "ijar_prebuilt_binary_or_cc_binary",
    actual = select({
        "//src/conditions:linux_x86_64": "@remote_java_tools_linux//:ijar_prebuilt_binary",
        "//src/conditions:darwin": "@remote_java_tools_darwin//:ijar_prebuilt_binary",
        "//src/conditions:windows": "@remote_java_tools_windows//:ijar_prebuilt_binary",
        "//conditions:default": "@remote_java_tools//:ijar_cc_binary",
    }),
)

alias(
    name = "ijar_prebuilt_binary",
    actual = select({
        "//src/conditions:linux_x86_64": "@remote_java_tools_linux//:ijar_prebuilt_binary",
        "//src/conditions:darwin": "@remote_java_tools_darwin//:ijar_prebuilt_binary",
        "//src/conditions:windows": "@remote_java_tools_windows//:ijar_prebuilt_binary",
    }),
)

# On Windows, Java implementation of singlejar is used. We create a little
# filegroup that contains the appropriate platform-dependent file.
# Once https://github.com/bazelbuild/bazel/issues/2241 is fixed (that is,
# the native singlejar is used on windows), this file group can be reused since
# on Windows, executables end in ".exe", but the label we reach it through
# must be platform-independent.
# Required by: Android, scala
alias(
    name = "singlejar",
    actual = select({
        ":remote": "@remote_java_tools//:singlejar_cc_bin",
        "//conditions:default": ":singlejar_prebuilt_or_cc_binary",
    }),
)

alias(
    name = "singlejar_prebuilt_or_cc_binary",
    actual = select({
        "//src/conditions:linux_x86_64": "@remote_java_tools_linux//:prebuilt_singlejar",
        "//src/conditions:darwin": "@remote_java_tools_darwin//:prebuilt_singlejar",
        "//src/conditions:windows": "@remote_java_tools_windows//:prebuilt_singlejar",
        "//conditions:default": "@remote_java_tools//:singlejar_cc_bin",
    }),
)

alias(
    name = "prebuilt_singlejar",
    actual = select({
        "//src/conditions:linux_x86_64": "@remote_java_tools_linux//:prebuilt_singlejar",
        "//src/conditions:darwin": "@remote_java_tools_darwin//:prebuilt_singlejar",
        "//src/conditions:windows": "@remote_java_tools_windows//:prebuilt_singlejar",
    }),
)

exports_files(["BUILD.java_tools"])

# Required by: default_java_toolchain, used by tests
alias(
    name = "genclass",
    actual = "@remote_java_tools//:GenClass",
)

alias(
    name = "GenClass_deploy.jar",
    actual = "@remote_java_tools//:GenClass",
)

alias(
    name = "bazel-singlejar_deploy.jar",
    actual = "@remote_java_tools//:SingleJar",
)

# Required by: default_java_toolchain, used by tests
alias(
    name = "turbine",
    actual = "@remote_java_tools//:Turbine",
)

alias(
    name = "turbine_deploy.jar",
    actual = "@remote_java_tools//:Turbine",
)

# Required by: default_java_toolchain, used by tests
alias(
    name = "turbine_direct",
    actual = "@remote_java_tools//:TurbineDirect",
)

alias(
    name = "turbine_direct_binary_deploy.jar",
    actual = "@remote_java_tools//:TurbineDirect",
)

# Required by: default_java_toolchain, used by tests
alias(
    name = "javabuilder",
    actual = "@remote_java_tools//:JavaBuilder",
)

# Required by: default_java_toolchain, used by tests
alias(
    name = "JavaBuilder_deploy.jar",
    actual = "@remote_java_tools//:JavaBuilder",
)

# Required by: 1 test directly, default_java_toolchain, used by tests
alias(
    name = "vanillajavabuilder",
    actual = "@remote_java_tools//:VanillaJavaBuilder",
)

# Required by: default_java_toolchain, used by tests
alias(
    name = "javac_jar",
    actual = "@remote_java_tools//:javac_jar",
)

# Required by: default_java_toolchain, used by tests
alias(
    name = "jdk_compiler_jar",
    actual = "@remote_java_tools//:jdk_compiler_jar",
)

# Required by: default_java_toolchain, used by tests
alias(
    name = "java_compiler_jar",
    actual = "@remote_java_tools//:java_compiler_jar",
)

# Required by: Bazel java_test, java_binary rule
alias(
    name = "JacocoCoverageRunner",
    actual = "@remote_java_tools//:jacoco_coverage_runner",
)

alias(
    name = "JacocoCoverage",
    actual = "@remote_java_tools//:jacoco_coverage_runner",
)

# Required by: Bazel java_test rule
java_import(
    name = "TestRunner",
    jars = ["@remote_java_tools//:Runner"],
)

# Required by: Android, scala
alias(
    name = "TestRunner_deploy.jar",
    actual = "@remote_java_tools//:Runner",
)

# Required by: Android: @bazel_tools//tools/android:build_java8_legacy_dex
alias(
    name = "proguard",
    actual = "@remote_java_tools//:proguard",
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

# Required by: legacy targets
alias(
    name = "jdk",
    actual = "@local_jdk//:jdk",
)

# TODO cleaner definition using toolchains
bootclasspath(
    name = "platformclasspath",
    src = "DumpPlatformClassPath.java",
    host_javabase = ":current_java_runtime",
    target_javabase = ":current_java_runtime",
)

default_java_toolchain(
    name = "toolchain_hostjdk8",
    configuration = JVM8_TOOLCHAIN_CONFIGURATION,
    source_version = "8",
    target_version = "8",
)

# Default to the Java 8 language level.
# TODO(cushon): consider if/when we should increment this?
# Required by ":toolchain"
default_java_toolchain(
    name = "legacy_toolchain",
    configuration = DEFAULT_TOOLCHAIN_CONFIGURATION,
    source_version = "8",
    target_version = "8",
)

# Required by: legacy targets
default_java_toolchain(
    name = "toolchain",
    configuration = DEFAULT_TOOLCHAIN_CONFIGURATION,
)

# Required by: legacy targets
alias(
    name = "remote_toolchain",
    actual = ":toolchain",
)

# Required by: 1 test
default_java_toolchain(
    name = "toolchain_vanilla",
    configuration = VANILLA_TOOLCHAIN_CONFIGURATION,
)

RELEASES = (8, 9, 10, 11)

[
    default_java_toolchain(
        name = "toolchain_java%d" % release,
        configuration = DEFAULT_TOOLCHAIN_CONFIGURATION,
        source_version = "%s" % release,
        target_version = "%s" % release,
    )
    for release in RELEASES
]

# A toolchain that targets java 14.
default_java_toolchain(
    name = "toolchain_jdk_14",
    configuration = dict(),
    source_version = "14",
    target_version = "14",
)

# A toolchain that targets java 15.
default_java_toolchain(
    name = "toolchain_jdk_15",
    configuration = dict(),
    source_version = "15",
    target_version = "15",
)

default_java_toolchain(
    name = "prebuilt_toolchain",
    configuration = PREBUILT_TOOLCHAIN_CONFIGURATION,
)

filegroup(
    name = "srcs",
    srcs = [
        "BUILD.java_langtools",
        "BUILD.java_tools",
        "BUILD.java_tools_prebuilt",
        "BUILD-jdk",  # Tools are build from the workspace for tests.
        "DumpPlatformClassPath.java",
        "default_java_toolchain.bzl",
        "fail_rule.bzl",
        "java_toolchain_alias.bzl",
        "jdk.BUILD",
        "local_java_repository.bzl",
        "nosystemjdk/README",
        "proguard_whitelister.py",
        "proguard_whitelister_test.py",
        "proguard_whitelister_test_input.pgcfg",
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

# A JDK 11 for use as a --host_javabase.
# Required by: legacy targets
alias(
    name = "remote_jdk11",
    actual = select(
        {
            "//src/conditions:darwin": "@remotejdk11_macos//:jdk",
            "//src/conditions:windows": "@remotejdk11_win//:jdk",
            "//src/conditions:linux_aarch64": "@remotejdk11_linux_aarch64//:jdk",
            "//src/conditions:linux_x86_64": "@remotejdk11_linux//:jdk",
            "//src/conditions:linux_ppc64le": "@remotejdk11_linux_ppc64le//:jdk",
            "//src/conditions:linux_s390x": "@remotejdk11_linux_s390x//:jdk",
        },
        no_match_error = "Could not find a JDK for host execution environment, please explicitly" +
                         " provide one using `--host_javabase.`",
    ),
    visibility = ["//visibility:public"],
)

# Java toolchains

[config_setting(
    name = "java_language_version_%s" % VER,
    values = {"java_language_version": VER},
) for VER in ("8", "9", "10", "11", "14", "15")]

OS_TOOLCHAIN_MAP = {
    "linux": "@remote_java_tools_linux//:toolchain",
    "freebsd": "@remote_java_tools_linux//:toolchain",
    "openbsd": "@remote_java_tools_linux//:toolchain",  #TODO different jvm_opts
    "windows": "@remote_java_tools_windows//:toolchain",
    "macos": "@remote_java_tools_darwin//:toolchain",
}

[
    toolchain(
        name = "remote_java_tools_%s" % os,
        exec_compatible_with = [
            "@platforms//os:%s" % os,
            "@platforms//cpu:x86_64",
        ],
        toolchain = select({
            "java_language_version_8": "%s_java8" % toolchain_label,
            "java_language_version_9": "%s_java9" % toolchain_label,
            "java_language_version_10": "%s_java10" % toolchain_label,
            "java_language_version_11": "%s_java11" % toolchain_label,
            "java_language_version_14": "%s_jdk_14" % toolchain_label,
            "java_language_version_15": "%s_jdk_15" % toolchain_label,
            "//conditions:default": "%s_java8" % toolchain_label,
        }),
        toolchain_type = ":toolchain_type",
    )
    for os, toolchain_label in OS_TOOLCHAIN_MAP.items()
]

# Java Runtime toolchains

[config_setting(
    name = "java_runtime_version_%s" % VER,
    values = {"java_runtime_version": VER},
) for VER in ("8", "9", "10", "11", "14")]

JDKS = {
    "linux": [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
    "macos": [
        "@platforms//os:macos",
        "@platforms//cpu:x86_64",
    ],
    "win": [
        "@platforms//os:windows",
        "@platforms//cpu:x86_64",
    ],
}

JDKS11_ONLY = {
    "remotejdk11_linux_aarch64": [
        "@platforms//os:linux",
        "@platforms//cpu:aarch64",
    ],
    "remotejdk11_linux_ppc64le": [
        "@platforms//os:linux",
        "@platforms//cpu:ppc",
    ],
    "remotejdk11_linux_s390x": [
        "@platforms//os:linux",
        "@platforms//cpu:s390x",
    ],
}

[
    (
        toolchain(
            name = "remotejdk_%s" % os,
            exec_compatible_with = constraints,
            toolchain =
                select({
                    "java_runtime_version_8": "@remotejdk11_%s//:jdk" % os,
                    "java_runtime_version_9": "@remotejdk11_%s//:jdk" % os,
                    "java_runtime_version_10": "@remotejdk11_%s//:jdk" % os,
                    "java_runtime_version_11": "@remotejdk11_%s//:jdk" % os,
                    "java_runtime_version_14": "@remotejdk14_%s//:jdk" % os,
                    "//conditions:default": ":jdk",
                }),
            toolchain_type = ":runtime_toolchain_type",
        ),
    )
    for os, constraints in JDKS.items()
]

[
    (
        toolchain(
            name = toolchain_name,
            exec_compatible_with = constraints,
            toolchain = "@%s//:jdk" % toolchain_name,
            toolchain_type = ":runtime_toolchain_type",
        ),
    )
    for toolchain_name, constraints in JDKS11_ONLY.items()
]
