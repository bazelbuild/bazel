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
    deps = [
        ":JarOwner",
        "//third_party:guava",
    ],
)

java_library(
    name = "javac",
    srcs = glob(
        [
            "javac/*.java",
        ],
        exclude = [
            "javac/JavacOptions.java",
        ],
    ),
    deps = [
        ":invalid_command_line_exception",
        ":javac_options",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/javac/plugins",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/java/jdk/langtools:javac",
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
        ":JarOwner",
        ":invalid_command_line_exception",
        ":javac",
        ":javac_options",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/jarhelper",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/javac/plugins",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/javac/plugins:dependency",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/javac/plugins:errorprone",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/javac/plugins:processing",
        "//src/main/protobuf:worker_protocol_java_proto",
        "//third_party:guava",
        "//third_party/java/jacoco:core",
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
    visibility = [
        "//java/com/google/devtools/build/buildjar:__pkg__",
        "//src/java_tools/buildjar/java/com/google/devtools/build/java/bazel:__pkg__",
        "//src/java_tools/buildjar/java/com/google/devtools/build/java/turbine/javac:__pkg__",
    ],
    deps = [
        "//third_party:auto_value",
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

java_library(
    name = "JarOwner",
    srcs = ["JarOwner.java"],
    visibility = [
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar:__subpackages__",
        "//src/java_tools/buildjar/java/com/google/devtools/build/java/turbine:__subpackages__",
        "//src/java_tools/buildjar/javatests/com/google/devtools/build/java/turbine:__subpackages__",
    ],
    deps = [
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

load("//tools/build_rules:java_rules_skylark.bzl", "bootstrap_java_library", "bootstrap_java_binary")

bootstrap_java_library(
    name = "skylark-deps",
    jars = [
        "//third_party:auto_value-jars",
        "//third_party:bootstrap_guava_and_error_prone-jars",
        "//third_party:jsr305-jars",
        "//third_party/protobuf:protobuf-jars",
        "//third_party/java/jacoco:core-jars",
    ],
    neverlink_jars = ["//third_party/java/jdk/langtools:javac_jar"],
    srcjars = [
        "//src/main/protobuf:deps_java_proto_srcjar",
        "//src/main/protobuf:worker_protocol_java_proto_srcjar",
        "//src/main/protobuf:java_compilation_java_proto_srcjar",
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
        exclude = [
            "java/com/google/devtools/build/buildjar/javac/testing/**",
            "JarOwner.java",
            "VanillaJavaBuilder.java",
        ],
    ),
    main_class = "com.google.devtools.build.buildjar.BazelJavaBuilder",
    deps = [
        ":skylark-deps",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/jarhelper:bootstrap_jarhelper",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/javac/plugins:bootstrap_plugins",
    ],
)

bootstrap_java_library(
    name = "bootstrap_JarOwner",
    srcs = ["JarOwner.java"],
    jars = [
        "//third_party:jsr305-jars",
        "//third_party:auto_value-jars",
        "//third_party:bootstrap_guava_and_error_prone-jars",
    ],
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
    deps = [
        ":skylark-deps",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/jarhelper:bootstrap_jarhelper",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/resourcejar:bootstrap_resourcejar",
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
        ":javac_options",
        ":optionsparser",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/jarhelper",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/resourcejar",
        "//src/main/protobuf:deps_java_proto",
        "//src/main/protobuf:java_compilation_java_proto",
        "//src/main/protobuf:worker_protocol_java_proto",
        "//third_party:guava",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/genclass:srcs",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/jarhelper:srcs",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/resourcejar:srcs",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/javac/plugins:srcs",
    ],
    visibility = ["//src/java_tools/buildjar:__pkg__"],
)
