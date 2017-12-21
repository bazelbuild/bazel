# Description:
# JavaBuilder and java tools used by Bazel
package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = [
        "BUILD",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar:srcs",
        "//src/java_tools/buildjar/java/com/google/devtools/build/java/bazel:srcs",
        "//src/java_tools/buildjar/java/com/google/devtools/build/java/turbine:srcs",
        "//src/java_tools/buildjar/javatests/com/google/devtools/build/buildjar:srcs",
        "//src/java_tools/buildjar/javatests/com/google/devtools/build/java/bazel:srcs",
        "//src/java_tools/buildjar/javatests/com/google/devtools/build/java/turbine:srcs",
    ],
)

java_binary(
    name = "JavaBuilder",
    main_class = "com.google.devtools.build.buildjar.BazelJavaBuilder",
    runtime_deps = ["//src/java_tools/buildjar/java/com/google/devtools/build/buildjar"],
)

java_binary(
    name = "VanillaJavaBuilder",
    main_class = "com.google.devtools.build.buildjar.VanillaJavaBuilder",
    runtime_deps = ["//src/java_tools/buildjar/java/com/google/devtools/build/buildjar:vanilla_java_builder"],
)

filegroup(
    name = "JavaBuilderDeploy",
    srcs = [":JavaBuilder_deploy.jar"],
    tags = ["manual"],
)

filegroup(
    name = "bootstrap_deploy.jar",
    srcs = ["//src/java_tools/buildjar/java/com/google/devtools/build/buildjar:bootstrap_deploy.jar"],
    tags = ["manual"],
)

filegroup(
    name = "bootstrap_VanillaJavaBuilder_deploy.jar",
    srcs = ["//src/java_tools/buildjar/java/com/google/devtools/build/buildjar:bootstrap_VanillaJavaBuilder_deploy.jar"],
    tags = ["manual"],
)

filegroup(
    name = "bootstrap_genclass_deploy.jar",
    srcs = ["//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/genclass:bootstrap_genclass_deploy.jar"],
    tags = ["manual"],
)

java_library(
    name = "BazelJavaCompiler",
    visibility = ["//visibility:public"],
    exports = [
        "//src/java_tools/buildjar/java/com/google/devtools/build/java/bazel:BazelJavaCompiler",
    ],
)

# This toolchain is used to bootstrap Bazel.
java_toolchain(
    name = "bootstrap_toolchain",
    # javac -extdirs is implemented by appending the contents to the platform
    # class path after -bootclasspath. For convenience, we currently have a
    # single jar that contains the contents of both the bootclasspath and
    # extdirs.
    bootclasspath = ["//tools/jdk:platformclasspath-impl.jar"],
    encoding = "UTF-8",
    extclasspath = [],
    genclass = ["bootstrap_genclass_deploy.jar"],
    ijar = ["//third_party/ijar"],
    javabuilder = ["bootstrap_deploy.jar"],
    javac = ["//third_party/java/jdk/langtools:javac_jar"],
    jvm_opts = [
        "-XX:+TieredCompilation",
        "-XX:TieredStopAtLevel=1",
        "-Xbootclasspath/p:$(location //third_party/java/jdk/langtools:javac_jar)",
    ],
    singlejar = ["//src/java_tools/singlejar:bootstrap_deploy.jar"],
    source_version = "8",
    tags = ["manual"],
    target_version = "8",
    visibility = ["//visibility:public"],
)
