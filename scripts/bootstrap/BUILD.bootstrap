load("@bazel_tools//tools/jdk:default_java_toolchain.bzl", "default_java_toolchain")

# This toolchain is used to bootstrap Bazel.
default_java_toolchain(
    name = "bootstrap_toolchain",
    bootclasspath = ["@bazel_tools//tools/jdk:platformclasspath"],
    genclass = ["//src/java_tools/buildjar:bootstrap_genclass_deploy.jar"],
    header_compiler = None,  # avoid remote_java_tools dependency
    header_compiler_direct = None,  # ditto
    ijar = ["//third_party/ijar"],
    jacocorunner = None,  # avoid remote_java_tools dependency
    java_runtime = "@local_jdk//:jdk",
    javabuilder = ["//src/java_tools/buildjar:bootstrap_VanillaJavaBuilder_deploy.jar"],
    jvm_opts = [
        # Prevent "Could not reserve enough space for object heap" errors on Windows.
        "-Xmx512m",
        # Using tiered compilation improves performance of Javac when not using the worker mode.
        "-XX:+TieredCompilation",
        "-XX:TieredStopAtLevel=1",
    ],
    singlejar = ["//src/tools/singlejar:singlejar"],
    source_version = "8",
    tags = ["manual"],
    target_version = "8",
    visibility = ["//visibility:public"],
)
