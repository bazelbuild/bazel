load("@rules_java//java:defs.bzl", "java_library")
load("//tools/build_rules:java_rules_skylark.bzl", "bootstrap_java_library")

# Description:
#   Plugins for the Java library builders, which are used by Bazel to
#   compile Java source code.
package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src/java_tools/buildjar:buildjar_package_group"],
)

java_library(
    name = "plugins",
    srcs = ["BlazeJavaCompilerPlugin.java"],
    javacopts = [
        "--add-exports=jdk.compiler/com.sun.tools.javac.comp=ALL-UNNAMED",
        "--add-exports=jdk.compiler/com.sun.tools.javac.main=ALL-UNNAMED",
        "--add-exports=jdk.compiler/com.sun.tools.javac.util=ALL-UNNAMED",
    ],
    deps = [
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar:invalid_command_line_exception",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/javac/statistics",
        "//third_party:guava",
    ],
)

java_library(
    name = "dependency",
    srcs = glob(["dependency/*.java"]),
    javacopts = [
        "--add-exports=jdk.compiler/com.sun.tools.javac.comp=ALL-UNNAMED",
        "--add-exports=jdk.compiler/com.sun.tools.javac.code=ALL-UNNAMED",
        "--add-exports=jdk.compiler/com.sun.tools.javac.file=ALL-UNNAMED",
        "--add-exports=jdk.compiler/com.sun.tools.javac.main=ALL-UNNAMED",
        "--add-exports=jdk.compiler/com.sun.tools.javac.resources=ALL-UNNAMED",
        "--add-exports=jdk.compiler/com.sun.tools.javac.tree=ALL-UNNAMED",
        "--add-exports=jdk.compiler/com.sun.tools.javac.util=ALL-UNNAMED",
    ],
    deps = [
        ":plugins",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar:JarOwner",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/javac/statistics",
        "//src/main/protobuf:deps_java_proto",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "javabuilder_errorprone_api",
    srcs = ["errorprone/ErrorProneInvoker.java"],
    visibility = ["//visibility:public"],
    deps = [
        ":plugins",
    ],
)

java_library(
    name = "errorprone",
    srcs = [
        "errorprone/BazelScannerSuppliers.java",
        "errorprone/ErrorProneInvokerImpl.java",
        "errorprone/ErrorPronePlugin.java",
    ],
    javacopts = [
        "--add-exports=jdk.compiler/com.sun.tools.javac.code=ALL-UNNAMED",
        "--add-exports=jdk.compiler/com.sun.tools.javac.comp=ALL-UNNAMED",
        "--add-exports=jdk.compiler/com.sun.tools.javac.main=ALL-UNNAMED",
        "--add-exports=jdk.compiler/com.sun.tools.javac.tree=ALL-UNNAMED",
        "--add-exports=jdk.compiler/com.sun.tools.javac.util=ALL-UNNAMED",
    ],
    resources = ["errorprone/META-INF/services/com.google.devtools.build.buildjar.javac.plugins.errorprone.ErrorProneInvoker"],
    resource_strip_prefix = "src/java_tools/buildjar/java/com/google/devtools/build/buildjar/javac/plugins/errorprone",
    visibility = ["//visibility:public"],
    deps = [
        ":javabuilder_errorprone_api",
        ":plugins",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar:invalid_command_line_exception",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/javac/statistics",
        "//third_party:error_prone",
        "//third_party:guava",
    ],
)

java_library(
    name = "processing",
    srcs = glob(["processing/*.java"]),
    javacopts = [
        "--add-exports=jdk.compiler/com.sun.tools.javac.comp=ALL-UNNAMED",
        "--add-exports=jdk.compiler/com.sun.tools.javac.tree=ALL-UNNAMED",
        "--add-exports=jdk.compiler/com.sun.tools.javac.util=ALL-UNNAMED",
    ],
    deps = [
        ":plugins",
        "//src/main/protobuf:java_compilation_java_proto",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**/*.java"]) + [
        "BUILD",
        "errorprone/META-INF/services/com.google.devtools.build.buildjar.javac.plugins.errorprone.ErrorProneInvoker",
    ],
    visibility = ["//src:__subpackages__"],
)

#
## Bootstrapping using Starlark rules
#

bootstrap_java_library(
    name = "bootstrap_plugins",
    srcs = glob([
        "dependency/*.java",
        "processing/*.java",
    ]),
    tags = [
        "manual",
        "nobuilder",
    ],
    deps = [
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar:bootstrap_JarOwner",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar:bootstrap_invalid_command_line_exception",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar:starlark-deps",
    ],
)
