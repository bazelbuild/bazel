package(default_visibility = ["//visibility:public"])

exports_files(glob(["**/*.jar"]))

licenses(["notice"])  # Apache 2.0

load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_proto_library")
load("@rules_java//java:defs.bzl", "java_binary", "java_import", "java_toolchain")
load("@rules_proto//proto:defs.bzl", "proto_library")
load(
    ":java_toolchain_default.bzl",
    "JAVABUILDER_TOOLCHAIN_CONFIGURATION",
    "JVM8_TOOLCHAIN_CONFIGURATION",
    "PREBUILT_TOOLCHAIN_CONFIGURATION",
    "VANILLA_TOOLCHAIN_CONFIGURATION",
    "java_toolchain_default",
)

SUPRESSED_WARNINGS = select({
    ":windows": [],
    "//conditions:default": [
        "-Wno-error",
        "-Wno-old-style-cast",
    ],
})

java_toolchain_default(
    name = "toolchain",
    configuration = JAVABUILDER_TOOLCHAIN_CONFIGURATION,
)

java_toolchain_default(
    name = "toolchain_hostjdk8",
    configuration = JVM8_TOOLCHAIN_CONFIGURATION,
    source_version = "8",
    target_version = "8",
)

# Default to the Java 8 language level.
# TODO(cushon): consider if/when we should increment this?
java_toolchain_default(
    name = "legacy_toolchain",
    configuration = JAVABUILDER_TOOLCHAIN_CONFIGURATION,
    source_version = "8",
    target_version = "8",
)

# Needed for openbsd / JVM8
java_toolchain_default(
    name = "legacy_toolchain_jvm8",
    configuration = JVM8_TOOLCHAIN_CONFIGURATION,
    source_version = "8",
    target_version = "8",
)

java_toolchain_default(
    name = "toolchain_vanilla",
    configuration = VANILLA_TOOLCHAIN_CONFIGURATION,
)

RELEASES = (8, 9, 10, 11)

[
    (
        java_toolchain_default(
            name = "toolchain_java%d" % release,
            configuration = JAVABUILDER_TOOLCHAIN_CONFIGURATION,
            source_version = "%s" % release,
            target_version = "%s" % release,
        ),
        # Needed for openbsd / JVM8
        java_toolchain_default(
            name = "toolchain_java%d_jvm8" % release,
            configuration = JVM8_TOOLCHAIN_CONFIGURATION,
            source_version = "%s" % release,
            target_version = "%s" % release,
        ),
    )
    for release in RELEASES
]

# A toolchain that targets java 14.
java_toolchain_default(
    name = "toolchain_jdk_14",
    source_version = "14",
    target_version = "14",
)

# A toolchain that targets java 15.
java_toolchain_default(
    name = "toolchain_jdk_15",
    source_version = "15",
    target_version = "15",
)

java_toolchain_default(
    name = "prebuilt_toolchain",
    configuration = PREBUILT_TOOLCHAIN_CONFIGURATION,
)

filegroup(
    name = "GenClass",
    srcs = ["java_tools/GenClass_deploy.jar"],
)

filegroup(
    name = "jacoco_coverage_runner_filegroup",
    srcs = ["java_tools/JacocoCoverage_jarjar_deploy.jar"],
)

java_import(
    name = "jacoco_coverage_runner",
    jars = ["java_tools/JacocoCoverage_jarjar_deploy.jar"],
)

filegroup(
    name = "JacocoCoverage",
    srcs = ["java_tools/JacocoCoverage_jarjar_deploy.jar"],
)

filegroup(
    name = "JavaBuilder",
    srcs = ["java_tools/JavaBuilder_deploy.jar"],
)

filegroup(
    name = "Runner",
    srcs = ["java_tools/Runner_deploy.jar"],
)

filegroup(
    name = "VanillaJavaBuilder",
    srcs = ["java_tools/VanillaJavaBuilder_deploy.jar"],
)

filegroup(
    name = "SingleJar",
    srcs = ["java_tools/bazel-singlejar_deploy.jar"],
)

filegroup(
    name = "Turbine",
    srcs = ["java_tools/turbine_deploy.jar"],
)

filegroup(
    name = "TurbineDirect",
    srcs = ["java_tools/turbine_direct_binary_deploy.jar"],
)

filegroup(
    name = "javac_jar",
    srcs = ["java_tools/javac-9+181-r4173-1.jar"],
)

filegroup(
    name = "jdk_compiler_jar",
    srcs = ["java_tools/jdk_compiler.jar"],
)

filegroup(
    name = "java_compiler_jar",
    srcs = ["java_tools/java_compiler.jar"],
)

java_import(
    name = "jacoco-agent",
    jars = ["java_tools/third_party/java/jacoco/org.jacoco.agent-0.8.3.jar"],
    srcjar = "java_tools/third_party/java/jacoco/org.jacoco.agent-0.8.3-src.jar",
)

java_import(
    name = "jacoco-core",
    jars = ["java_tools/third_party/java/jacoco/org.jacoco.core-0.8.3.jar"],
    srcjar = "java_tools/third_party/java/jacoco/org.jacoco.core-0.8.3-src.jar",
    exports = [
        ":asm",
        ":asm-commons",
        ":asm-tree",
    ],
)

filegroup(
    name = "jacoco-core-jars",
    srcs = ["java_tools/third_party/java/jacoco/org.jacoco.core-0.8.3.jar"],
)

java_import(
    name = "jacoco-report",
    jars = ["java_tools/third_party/java/jacoco/org.jacoco.report-0.8.3.jar"],
    srcjar = "java_tools/third_party/java/jacoco/org.jacoco.report-sources.jar",
    exports = [
        ":asm",
        ":jacoco-core",
    ],
)

java_import(
    name = "bazel-jacoco-agent",
    jars = ["java_tools/third_party/java/jacoco/jacocoagent-0.8.3.jar"],
)

java_import(
    name = "bazel-jacoco-agent-neverlink",
    jars = ["java_tools/third_party/java/jacoco/jacocoagent-0.8.3.jar"],
    neverlink = 1,
)

java_import(
    name = "asm",
    jars = ["java_tools/third_party/java/jacoco/asm-8.0.jar"],
    srcjar = "java_tools/third_party/java/jacoco/asm-8.0-sources.jar",
)

java_import(
    name = "asm-commons",
    jars = ["java_tools/third_party/java/jacoco/asm-commons-8.0.jar"],
    srcjar = "java_tools/third_party/java/jacoco/asm-commons-8.0-sources.jar",
    runtime_deps = [":asm-tree"],
)

java_import(
    name = "asm-tree",
    jars = ["java_tools/third_party/java/jacoco/asm-tree-8.0.jar"],
    srcjar = "java_tools/third_party/java/jacoco/asm-tree-8.0-sources.jar",
    runtime_deps = [":asm"],
)

config_setting(
    name = "remote",
    values = {"define": "EXECUTOR=remote"},
)

config_setting(
    name = "linux_x86_64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
)

config_setting(
    name = "darwin_x86_64",
    constraint_values = [
        "@platforms//os:macos",
        "@platforms//cpu:x86_64",
    ],
)

config_setting(
    name = "darwin_arm64",
    constraint_values = [
        "@platforms//os:macos",
        "@platforms//cpu:arm64",
    ],
)

config_setting(
    name = "darwin_arm64e",
    constraint_values = [
        "@platforms//os:macos",
        "@platforms//cpu:arm64e",
    ],
)

config_setting(
    name = "windows",
    constraint_values = ["@platforms//os:windows"],
)

config_setting(
    name = "freebsd",
    constraint_values = ["@platforms//os:freebsd"],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "openbsd",
    constraint_values = ["@platforms//os:openbsd"],
    visibility = ["//visibility:public"],
)

alias(
    name = "singlejar",
    actual = select({
        "//:remote": ":singlejar_cc_bin",
        "//conditions:default": ":singlejar_prebuilt_or_cc_binary",
    }),
)

alias(
    name = "singlejar_prebuilt_or_cc_binary",
    actual = select({
        ":linux_x86_64": "java_tools/src/tools/singlejar/singlejar_local",
        ":darwin_x86_64": "java_tools/src/tools/singlejar/singlejar_local",
        ":darwin_arm64": "java_tools/src/tools/singlejar/singlejar_local",
        ":darwin_arm64e": "java_tools/src/tools/singlejar/singlejar_local",
        ":windows": "java_tools/src/tools/singlejar/singlejar_local.exe",
        "//conditions:default": "singlejar_cc_bin",
    }),
)

alias(
    name = "prebuilt_singlejar",
    actual = select({
        ":windows": "java_tools/src/tools/singlejar/singlejar_local.exe",
        "//conditions:default": "java_tools/src/tools/singlejar/singlejar_local",
    }),
)

alias(
    name = "ijar",
    actual = select({
        ":remote": ":ijar_cc_binary",
        "//conditions:default": ":prebuilt_binary_or_cc_binary",
    }),
)

alias(
    name = "prebuilt_binary_or_cc_binary",
    actual = select({
        ":linux_x86_64": ":ijar_prebuilt_binary",
        ":darwin_x86_64": ":ijar_prebuilt_binary",
        ":darwin_arm64": ":ijar_prebuilt_binary",
        ":darwin_arm64e": ":ijar_prebuilt_binary",
        ":windows": ":ijar_prebuilt_binary",
        "//conditions:default": ":ijar_cc_binary",
    }),
)

filegroup(
    name = "ijar_prebuilt_binary",
    srcs = select({
        ":windows": ["java_tools/ijar/ijar.exe"],
        "//conditions:default": ["java_tools/ijar/ijar"],
    }),
)

cc_library(
    name = "malloc",
)

cc_binary(
    name = "ijar_cc_binary",
    srcs = [
        "java_tools/ijar/classfile.cc",
        "java_tools/ijar/ijar.cc",
    ],
    copts = SUPRESSED_WARNINGS,
    # Remove dependency on @bazel_tools//tools/cpp:malloc, which avoid /Iexternal/tools being used
    # in compiling actions.
    malloc = ":malloc",
    deps = [":zip"],
)

cc_library(
    name = "zip",
    srcs = [
        "java_tools/ijar/zip.cc",
    ] + select({
        ":windows": [
            "java_tools/ijar/mapped_file_windows.cc",
        ],
        "//conditions:default": [
            "java_tools/ijar/mapped_file_unix.cc",
        ],
    }),
    hdrs = [
        "java_tools/ijar/common.h",
        "java_tools/ijar/mapped_file.h",
        "java_tools/ijar/zip.h",
    ],
    copts = SUPRESSED_WARNINGS,
    include_prefix = "third_party",
    strip_include_prefix = "java_tools",
    deps = [
        ":platform_utils",
        ":zlib_client",
    ] + select({
        ":windows": [
            ":errors",
            ":filesystem",
            ":logging",
            ":strings",
        ],
        "//conditions:default": [
        ],
    }),
)

cc_library(
    name = "platform_utils",
    srcs = ["java_tools/ijar/platform_utils.cc"],
    hdrs = [
        "java_tools/ijar/common.h",
        "java_tools/ijar/platform_utils.h",
    ],
    copts = SUPRESSED_WARNINGS,
    include_prefix = "third_party",
    strip_include_prefix = "java_tools",
    visibility = ["//visibility:private"],
    deps = [
        ":errors",
        ":filesystem",
        ":logging",
    ],
)

cc_library(
    name = "cpp_util",
    hdrs = [
        "java_tools/src/main/cpp/util/errors.h",
        "java_tools/src/main/cpp/util/file.h",
        "java_tools/src/main/cpp/util/file_platform.h",
        "java_tools/src/main/cpp/util/md5.h",
        "java_tools/src/main/cpp/util/numbers.h",
        "java_tools/src/main/cpp/util/path.h",
        "java_tools/src/main/cpp/util/path_platform.h",
        "java_tools/src/main/cpp/util/port.h",
    ],
    strip_include_prefix = "java_tools",
    visibility = ["//visibility:public"],
    deps = [
        ":blaze_exit_code",
        ":errors",
        ":filesystem",
        ":md5",
        ":numbers",
        ":port",
        ":strings",
    ],
)

cc_library(
    name = "md5",
    srcs = ["java_tools/src/main/cpp/util/md5.cc"],
    hdrs = ["java_tools/src/main/cpp/util/md5.h"],
    strip_include_prefix = "java_tools",
)

cc_library(
    name = "numbers",
    srcs = ["java_tools/src/main/cpp/util/numbers.cc"],
    hdrs = ["java_tools/src/main/cpp/util/numbers.h"],
    strip_include_prefix = "java_tools",
    deps = [":strings"],
)

cc_library(
    name = "filesystem",
    srcs = [
        "java_tools/src/main/cpp/util/file.cc",
        "java_tools/src/main/cpp/util/path.cc",
    ] + select({
        ":windows": [
            "java_tools/src/main/cpp/util/file_windows.cc",
            "java_tools/src/main/cpp/util/path_windows.cc",
        ],
        "//conditions:default": [
            "java_tools/src/main/cpp/util/file_posix.cc",
            "java_tools/src/main/cpp/util/path_posix.cc",
        ],
    }),
    hdrs = [
        "java_tools/src/main/cpp/util/file.h",
        "java_tools/src/main/cpp/util/file_platform.h",
        "java_tools/src/main/cpp/util/path.h",
        "java_tools/src/main/cpp/util/path_platform.h",
    ],
    strip_include_prefix = "java_tools",
    deps = [
        ":blaze_exit_code",
        ":errors",
        ":logging",
        ":strings",
    ] + select({
        ":windows": [":lib-file"],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "lib-file",
    srcs = [
        "java_tools/src/main/native/windows/file.cc",
        "java_tools/src/main/native/windows/util.cc",
    ],
    hdrs = [
        "java_tools/src/main/native/windows/file.h",
        "java_tools/src/main/native/windows/util.h",
    ],
    strip_include_prefix = "java_tools",
)

cc_library(
    name = "errors",
    srcs = select({
        ":windows": ["java_tools/src/main/cpp/util/errors_windows.cc"],
        "//conditions:default": ["java_tools/src/main/cpp/util/errors_posix.cc"],
    }),
    hdrs = ["java_tools/src/main/cpp/util/errors.h"],
    strip_include_prefix = "java_tools",
    deps = [
        ":logging",
        ":port",
        ":strings",
    ],
)

cc_library(
    name = "strings",
    srcs = ["java_tools/src/main/cpp/util/strings.cc"],
    hdrs = ["java_tools/src/main/cpp/util/strings.h"],
    copts = SUPRESSED_WARNINGS,
    # Automatically propagate the symbol definition to rules depending on this.
    defines = [
        "BLAZE_OPENSOURCE",
    ],
    strip_include_prefix = "java_tools",
    deps = [":blaze_exit_code"],
)

cc_library(
    name = "blaze_exit_code",
    hdrs = ["java_tools/src/main/cpp/util/exit_code.h"],
    copts = SUPRESSED_WARNINGS,
    strip_include_prefix = "java_tools",
)

cc_library(
    name = "port",
    srcs = ["java_tools/src/main/cpp/util/port.cc"],
    hdrs = ["java_tools/src/main/cpp/util/port.h"],
    copts = SUPRESSED_WARNINGS,
    strip_include_prefix = "java_tools",
)

cc_library(
    name = "logging",
    srcs = ["java_tools/src/main/cpp/util/logging.cc"],
    hdrs = ["java_tools/src/main/cpp/util/logging.h"],
    copts = SUPRESSED_WARNINGS,
    strip_include_prefix = "java_tools",
    deps = [
        ":blaze_exit_code",
        ":strings",
    ],
)

cc_library(
    name = "zlib_client",
    srcs = ["java_tools/ijar/zlib_client.cc"],
    hdrs = [
        "java_tools/ijar/common.h",
        "java_tools/ijar/zlib_client.h",
    ],
    copts = SUPRESSED_WARNINGS,
    include_prefix = "third_party",
    strip_include_prefix = "java_tools",
    deps = ["//java_tools/zlib"],
)

##################### singlejar

cc_binary(
    name = "singlejar_cc_bin",
    srcs = [
        "java_tools/src/tools/singlejar/singlejar_main.cc",
    ],
    copts = SUPRESSED_WARNINGS,
    linkopts = select({
        ":freebsd": ["-lm"],
        ":openbsd": ["-lm"],
        "//conditions:default": [],
    }),
    linkstatic = 1,
    # Remove dependency on @bazel_tools//tools/cpp:malloc, which avoid /Iexternal/tools being used
    # in compiling actions.
    malloc = ":malloc",
    visibility = ["//visibility:public"],
    deps = [
        ":options",
        ":output_jar",
        "//java_tools/zlib",
    ],
)

cc_binary(
    name = "singlejar_local",
    srcs = [
        "java_tools/src/tools/singlejar/singlejar_local_main.cc",
    ],
    copts = SUPRESSED_WARNINGS,
    linkopts = select({
        ":freebsd": ["-lm"],
        ":openbsd": ["-lm"],
        "//conditions:default": [],
    }),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":combiners",
        ":desugar_checking",
        ":options",
        ":output_jar",
        "//java_tools/zlib",
    ],
)

cc_library(
    name = "combiners",
    srcs = [
        "java_tools/src/tools/singlejar/combiners.cc",
    ],
    hdrs = [
        "java_tools/src/tools/singlejar/combiners.h",
        ":transient_bytes",
    ],
    copts = SUPRESSED_WARNINGS,
    strip_include_prefix = "java_tools",
    deps = [
        "//java_tools/zlib",
    ],
)

proto_library(
    name = "desugar_deps_proto",
    srcs = ["java_tools/src/main/protobuf/desugar_deps.proto"],
)

cc_proto_library(
    name = "desugar_deps_cc_proto",
    deps = [":desugar_deps_proto"],
)

cc_library(
    name = "desugar_checking",
    srcs = ["java_tools/src/tools/singlejar/desugar_checking.cc"],
    hdrs = ["java_tools/src/tools/singlejar/desugar_checking.h"],
    copts = SUPRESSED_WARNINGS,
    strip_include_prefix = "java_tools",
    deps = [
        ":combiners",
        ":desugar_deps_cc_proto",
    ],
)

cc_library(
    name = "diag",
    hdrs = ["java_tools/src/tools/singlejar/diag.h"],
    copts = SUPRESSED_WARNINGS,
    strip_include_prefix = "java_tools",
    visibility = ["//visibility:private"],
)

cc_library(
    name = "singlejar_port",
    hdrs = ["java_tools/src/tools/singlejar/port.h"],
    copts = SUPRESSED_WARNINGS,
    strip_include_prefix = "java_tools",
    visibility = ["//visibility:private"],
)

cc_library(
    name = "mapped_file",
    srcs = ["java_tools/src/tools/singlejar/mapped_file.cc"],
    hdrs = ["java_tools/src/tools/singlejar/mapped_file.h"] +
           select({
               ":windows": ["java_tools/src/tools/singlejar/mapped_file_windows.inc"],
               "//conditions:default": ["java_tools/src/tools/singlejar/mapped_file_posix.inc"],
           }),
    copts = SUPRESSED_WARNINGS,
    strip_include_prefix = "java_tools",
    visibility = ["//visibility:private"],
    deps = [
        ":cpp_util",
        ":diag",
        ":singlejar_port",
    ],
)

cc_library(
    name = "input_jar",
    srcs = [
        "java_tools/src/tools/singlejar/input_jar.cc",
    ],
    hdrs = [
        "java_tools/src/tools/singlejar/input_jar.h",
        "java_tools/src/tools/singlejar/zip_headers.h",
    ],
    copts = SUPRESSED_WARNINGS,
    strip_include_prefix = "java_tools",
    deps = [
        ":diag",
        ":mapped_file",
    ],
)

cc_library(
    name = "options",
    srcs = [
        "java_tools/src/tools/singlejar/options.cc",
        "java_tools/src/tools/singlejar/options.h",
    ],
    hdrs = ["java_tools/src/tools/singlejar/options.h"],
    copts = SUPRESSED_WARNINGS,
    strip_include_prefix = "java_tools",
    deps = [
        ":diag",
        ":token_stream",
    ],
)

cc_library(
    name = "output_jar",
    srcs = [
        "java_tools/src/tools/singlejar/output_jar.cc",
        "java_tools/src/tools/singlejar/output_jar.h",
        ":zip_headers",
    ],
    hdrs = ["java_tools/src/tools/singlejar/output_jar.h"],
    copts = SUPRESSED_WARNINGS,
    strip_include_prefix = "java_tools",
    deps = [
        ":combiners",
        ":cpp_util",
        ":diag",
        ":input_jar",
        ":mapped_file",
        ":options",
        ":singlejar_port",
        "//java_tools/zlib",
    ],
)

cc_library(
    name = "token_stream",
    hdrs = ["java_tools/src/tools/singlejar/token_stream.h"],
    copts = SUPRESSED_WARNINGS,
    strip_include_prefix = "java_tools",
    deps = [
        ":diag",
        ":filesystem",
    ],
)

filegroup(
    name = "transient_bytes",
    srcs = [
        "java_tools/src/tools/singlejar/diag.h",
        "java_tools/src/tools/singlejar/transient_bytes.h",
        "java_tools/src/tools/singlejar/zlib_interface.h",
        ":zip_headers",
    ],
)

filegroup(
    name = "zip_headers",
    srcs = ["java_tools/src/tools/singlejar/zip_headers.h"],
)

################### Proguard ###################
java_import(
    name = "proguard_import",
    jars = ["java_tools/third_party/java/proguard/proguard.jar"],
)

java_binary(
    name = "proguard",
    main_class = "proguard.ProGuard",
    visibility = ["//visibility:public"],
    runtime_deps = [":proguard_import"],
)
