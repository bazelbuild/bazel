# Description:
# JavaBuilder and java tools used by Bazel
package(default_visibility = [":buildjar_package_group"])

filegroup(
    name = "srcs",
    srcs = [
        "BUILD",
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar:srcs",
        "//src/java_tools/buildjar/java/com/google/devtools/build/java/bazel:srcs",
        "//src/java_tools/buildjar/java/com/google/devtools/build/java/turbine:srcs",
        "//src/java_tools/buildjar/javatests/com/google/devtools/build/buildjar:srcs",
        "//src/java_tools/buildjar/javatests/com/google/devtools/build/java/turbine:srcs",
    ],
    visibility = ["//:__subpackages__"],
)

package_group(
    name = "buildjar_package_group",
    packages = [
        "//src/java_tools/buildjar/...",
        "//src/tools/android/...",
    ],
)

java_binary(
    name = "JavaBuilder",
    main_class = "com.google.devtools.build.buildjar.BazelJavaBuilder",
    visibility = ["//visibility:public"],
    runtime_deps = ["//src/java_tools/buildjar/java/com/google/devtools/build/buildjar"],
)

java_binary(
    name = "VanillaJavaBuilder",
    main_class = "com.google.devtools.build.buildjar.VanillaJavaBuilder",
    visibility = ["//visibility:public"],
    runtime_deps = ["//src/java_tools/buildjar/java/com/google/devtools/build/buildjar:vanilla_java_builder"],
)

filegroup(
    name = "JavaBuilderDeploy",
    srcs = [":JavaBuilder_deploy.jar"],
    tags = ["manual"],
    visibility = [
        ":buildjar_package_group",
        "//src:__pkg__",
    ],
)

filegroup(
    name = "bootstrap_VanillaJavaBuilder_deploy.jar",
    srcs = ["//src/java_tools/buildjar/java/com/google/devtools/build/buildjar:bootstrap_VanillaJavaBuilder_deploy.jar"],
    tags = ["manual"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "bootstrap_genclass_deploy.jar",
    srcs = ["//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/genclass:bootstrap_genclass_deploy.jar"],
    tags = ["manual"],
    visibility = ["//visibility:public"],
)

# This toolchain is used to bootstrap Bazel.
java_toolchain(
    name = "bootstrap_toolchain",
    # javac -extdirs is implemented by appending the contents to the platform
    # class path after -bootclasspath. For convenience, we currently have a
    # single jar that contains the contents of both the bootclasspath and
    # extdirs.
    bootclasspath = ["//tools/jdk:platformclasspath.jar"],
    extclasspath = [],
    genclass = ["bootstrap_genclass_deploy.jar"],
    ijar = ["//third_party/ijar"],
    javabuilder = ["bootstrap_VanillaJavaBuilder_deploy.jar"],
    javac = ["//third_party/java/jdk/langtools:javac_jar"],
    jvm_opts = [
        # Prevent "Could not reserve enough space for object heap" errors on Windows.
        "-Xmx512m",
        # Using tiered compilation improves performance of Javac when not using the worker mode.
        "-XX:+TieredCompilation",
        "-XX:TieredStopAtLevel=1",
    ],
    singlejar = ["//src/java_tools/singlejar:bootstrap_deploy.jar"],
    source_version = "8",
    tags = ["manual"],
    target_version = "8",
    visibility = ["//visibility:public"],
)
