# Description:
#   Plugins for the Java library builders, which are used by Bazel to
#   compile Java source code.

package_group(
    name = "packages",
    packages = [
        "//src/java_tools/buildjar/...",
    ],
)

package(default_visibility = [":packages"])

java_library(
    name = "plugins",
    srcs = ["BlazeJavaCompilerPlugin.java"],
    deps = [
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar:invalid_command_line_exception",
        "//third_party/java/jdk/langtools:javac",
    ],
)

java_library(
    name = "dependency",
    srcs = glob(["dependency/*.java"]),
    deps = [
        ":plugins",
        "//src/main/protobuf:deps_java_proto",
        "//third_party:guava",
        "//third_party/java/jdk/langtools:javac",
    ],
)

java_library(
    name = "errorprone",
    srcs = glob(["errorprone/*.java"]),
    deps = [
        ":plugins",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar:invalid_command_line_exception",
        "//third_party:error_prone",
        "//third_party:guava",
        "//third_party/java/jdk/langtools:javac",
    ],
)

java_library(
    name = "processing",
    srcs = glob(["processing/*.java"]),
    deps = [
        ":plugins",
        "//src/main/protobuf:java_compilation_java_proto",
        "//third_party:guava",
        "//third_party/java/jdk/langtools:javac",
    ],
)

java_library(
    name = "classloader",
    srcs = glob(["classloader/*.java"]),
    deps = [
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/javac/plugins",
        "//third_party:guava",
        "//third_party/java/jdk/langtools:javac",
    ],
)

#
## Bootstrapping using Skylark rules
#

load("//tools/build_rules:java_rules_skylark.bzl", "bootstrap_java_library", "bootstrap_java_binary")

bootstrap_java_library(
    name = "bootstrap_plugins",
    srcs = glob(
        ["**/*.java"],
    ),
    deps = [
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar:bootstrap_invalid_command_line_exception",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar:skylark-deps",
    ],
)
