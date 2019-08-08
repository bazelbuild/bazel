# Description:
#   The Java library builders, which are used by Bazel to compile Java
#   source code.
package(default_visibility = ["//src/java_tools/buildjar:buildjar_package_group"])

filegroup(
    name = "srcs",
    srcs = glob(
        ["**"],
    ) + [
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/javac/plugins:srcs",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/javac/statistics:srcs",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/genclass:srcs",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/jarhelper:srcs",
    ],
)

java_library(
    name = "optionsparser",
    srcs = [
        "InvalidCommandLineException.java",
        "OptionsParser.java",
    ],
    visibility = [
        "//src/java_tools/buildjar:buildjar_package_group",
        "//src/test/java/com/google/devtools/build/lib:__subpackages__",
    ],
    deps = [
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "javac",
    srcs = glob(
        ["javac/*.java"],
        exclude = [
            "javac/JavacOptions.java",
        ],
    ),
    deps = [
        ":invalid_command_line_exception",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/javac/plugins",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/javac/statistics",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/java/jdk/langtools:javac",
    ],
)

alias(
    name = "jarhelper",
    actual = "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/jarhelper",
    visibility = [
        "//src/java_tools/buildjar:buildjar_package_group",
        "//src/tools/android:__subpackages__",
    ],
)

# Bazel's BuildJar
java_library(
    name = "buildjar",
    srcs = glob(
        [
            "*.java",
            "instrumentation/*.java",
        ],
        exclude = [
            "InvalidCommandLineException.java",
            "JarOwner.java",
            "VanillaJavaBuilder.java",
        ],
    ),
    deps = [
        ":invalid_command_line_exception",
        ":jarhelper",
        ":javac",
        ":javac_options",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/javac/plugins",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/javac/plugins:dependency",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/javac/plugins:errorprone",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/javac/plugins:processing",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/javac/statistics",
        "//src/main/protobuf:worker_protocol_java_proto",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/java/jacoco:core",
        "//third_party/java/jdk/langtools:javac",
    ],
)

java_binary(
    name = "BazelJavaBuilder",
    main_class = "com.google.devtools.build.buildjar.BazelJavaBuilder",
    runtime_deps = [":buildjar"],
)

java_library(
    name = "javac_options",
    srcs = [
        "javac/JavacOptions.java",
    ],
    deps = [
        "//third_party:auto_value",
        "//third_party:guava",
    ],
)

java_library(
    name = "invalid_command_line_exception",
    srcs = ["InvalidCommandLineException.java"],
    deps = [],
)

java_library(
    name = "JarOwner",
    srcs = ["JarOwner.java"],
    deps = [
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_binary(
    name = "VanillaJavaBuilder",
    main_class = "com.google.devtools.build.buildjar.VanillaJavaBuilder",
    runtime_deps = [":vanilla_java_builder"],
)

java_library(
    name = "vanilla_java_builder",
    srcs = ["VanillaJavaBuilder.java"],
    deps = [
        ":jarhelper",
        ":javac_options",
        ":optionsparser",
        "//src/main/protobuf:deps_java_proto",
        "//src/main/protobuf:java_compilation_java_proto",
        "//src/main/protobuf:worker_protocol_java_proto",
        "//third_party:guava",
    ],
)

#
## Bootstrapping using Skylark rules
#

load("//tools/build_rules:java_rules_skylark.bzl", "bootstrap_java_library", "bootstrap_java_binary")

bootstrap_java_library(
    name = "skylark-deps",
    srcs = ["//:bootstrap-derived-java-srcs"],
    jars = [
        "//third_party:auto_value-jars",
        "//third_party:bootstrap_guava_and_error_prone-jars",
        "//third_party:jsr305-jars",
        "//third_party/protobuf:protobuf-jars",
        "//third_party/java/jacoco:core-jars",
        "//third_party/grpc:bootstrap-grpc-jars",
        "//third_party:tomcat_annotations_api-jars",
    ],
    neverlink_jars = ["//third_party/java/jdk/langtools:javac_jar"],
    tags = ["manual"],
)

bootstrap_java_library(
    name = "bootstrap_invalid_command_line_exception",
    srcs = ["InvalidCommandLineException.java"],
    tags = ["manual"],
)

bootstrap_java_library(
    name = "bootstrap_JarOwner",
    srcs = ["JarOwner.java"],
    jars = [
        "//third_party:jsr305-jars",
        "//third_party:auto_value-jars",
        "//third_party:bootstrap_guava_and_error_prone-jars",
    ],
    tags = ["manual"],
)

bootstrap_java_binary(
    name = "bootstrap_VanillaJavaBuilder",
    srcs = [
        "InvalidCommandLineException.java",
        "JarOwner.java",
        "OptionsParser.java",
        "VanillaJavaBuilder.java",
        "javac/JavacOptions.java",
    ],
    main_class = "com.google.devtools.build.buildjar.VanillaJavaBuilder",
    tags = ["manual"],
    deps = [
        ":skylark-deps",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/jarhelper:bootstrap_jarhelper",
    ],
)
