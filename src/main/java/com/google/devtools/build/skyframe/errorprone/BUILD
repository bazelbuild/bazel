load("@rules_java//java:defs.bzl", "java_library", "java_plugin")

package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

# Error Prone is provided by the compiler that loads the plugin.
java_library(
    name = "error_prone_neverlink",
    neverlink = True,
    exports = [
        "@maven//:com_google_errorprone_error_prone_annotation",
        "@maven//:com_google_errorprone_error_prone_check_api",
    ],
)

# Error Prone checks for Skyframe values, discovered via ServiceLoader on the processor path.
java_plugin(
    name = "errorprone",
    srcs = glob(["*.java"]),
    javacopts = [
        "--add-exports=jdk.compiler/com.sun.tools.javac.code=ALL-UNNAMED",
        "--add-exports=jdk.compiler/com.sun.tools.javac.tree=ALL-UNNAMED",
        "--add-exports=jdk.compiler/com.sun.tools.javac.util=ALL-UNNAMED",
    ],
    resource_strip_prefix = package_name(),
    resources = ["META-INF/services/com.google.errorprone.bugpatterns.BugChecker"],
    deps = [":error_prone_neverlink"],
)
