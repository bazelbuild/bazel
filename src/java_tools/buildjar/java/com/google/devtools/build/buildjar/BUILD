# Description:
#   The Java library builders, which are used by Bazel to compile Java
#   source code.

package_group(
    name = "packages",
    packages = [
        "//src/java_tools/buildjar/...",
    ],
)

package(default_visibility = [":packages"])

java_library(
    name = "optionsparser",
    srcs = [
        "InvalidCommandLineException.java",
        "OptionsParser.java",
    ],
)

java_library(
    name = "javac",
    srcs = glob([
        "javac/*.java",
    ]),
    deps = [
        ":invalid_command_line_exception",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/javac/plugins",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/java/jdk/langtools:javac",
    ],
)

# Bazel's BuildJar
java_library(
    name = "buildjar",
    srcs = glob(
        ["*.java"],
        exclude = [
            "InvalidCommandLineException.java",
        ],
    ),
    deps = [
        ":invalid_command_line_exception",
        ":javac",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/jarhelper",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/javac/plugins",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/javac/plugins:classloader",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/javac/plugins:dependency",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/javac/plugins:errorprone",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/javac/plugins:processing",
        "//src/main/protobuf:worker_protocol_proto",
        "//third_party:guava",
        "//third_party/java/jdk/langtools:javac",
    ],
)

java_binary(
    name = "BazelJavaBuilder",
    main_class = "com.google.devtools.build.buildjar.BazelJavaBuilder",
    visibility = ["//visibility:public"],
    runtime_deps = [":buildjar"],
)

java_library(
    name = "javac_options",
    srcs = [
        "javac/JavacOptions.java",
    ],
    visibility = ["//java/com/google/devtools/build/buildjar:__pkg__"],
    deps = [
        "//third_party:guava",
    ],
)

java_library(
    name = "invalid_command_line_exception",
    srcs = ["InvalidCommandLineException.java"],
    visibility = [
        "//java/com/google/devtools/build/buildjar:__subpackages__",
        "//javatests/com/google/devtools/build/buildjar:__subpackages__",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar:__subpackages__",
    ],
    deps = [],
)

load("//tools:build_rules/java_rules_skylark.bzl", "bootstrap_java_library", "bootstrap_java_binary")

bootstrap_java_library(
    name = "skylark-deps",
    jars = [
        "//third_party:error_prone-jars",
        "//third_party:guava-jars",
        "//third_party:jsr305-jars",
        "//third_party:protobuf-jars",
    ],
    neverlink_jars = ["//tools/defaults:java_langtools"],
    srcjars = [
        "//src/main/protobuf:deps_proto_srcjar",
        "//src/main/protobuf:worker_protocol_proto_srcjar",
        "//src/main/protobuf:java_compilation_proto_srcjar",
    ],
)

bootstrap_java_library(
    name = "bootstrap_invalid_command_line_exception",
    srcs = ["InvalidCommandLineException.java"],
)

bootstrap_java_binary(
    name = "bootstrap",
    srcs = glob(
        ["**/*.java"],
        exclude = ["java/com/google/devtools/build/buildjar/javac/testing/**"],
    ),
    main_class = "com.google.devtools.build.buildjar.BazelJavaBuilder",
    deps = [
        ":skylark-deps",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/jarhelper:bootstrap_jarhelper",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/javac/plugins:bootstrap_plugins",
    ],
)
