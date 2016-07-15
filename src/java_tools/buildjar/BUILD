package(default_visibility = ["//src:__subpackages__"])

java_binary(
    name = "JavaBuilder",
    main_class = "com.google.devtools.build.buildjar.BazelJavaBuilder",
    visibility = ["//visibility:public"],
    runtime_deps = ["//src/java_tools/buildjar/java/com/google/devtools/build/buildjar"],
)

filegroup(
    name = "JavaBuilderDeploy",
    srcs = select({
        "//tools/jdk:jdk7": ["//third_party/java/jdk/javabuilder:JavaBuilder_0.1.0"],
        "//conditions:default": [":JavaBuilder_deploy.jar"],
    }),
)

filegroup(
    name = "bootstrap_deploy.jar",
    srcs = ["//src/java_tools/buildjar/java/com/google/devtools/build/buildjar:bootstrap_deploy.jar"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "bootstrap_genclass_deploy.jar",
    srcs = ["//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/genclass:bootstrap_genclass_deploy.jar"],
    visibility = ["//visibility:public"],
)

java_library(
    name = "BazelJavaCompiler",
    visibility = ["//visibility:public"],
    exports = [
        "//src/java_tools/buildjar/java/com/google/devtools/build/java/bazel:BazelJavaCompiler",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar:srcs",
        "//src/java_tools/buildjar/java/com/google/devtools/build/java/bazel:srcs",
        "//src/java_tools/buildjar/java/com/google/devtools/build/java/turbine:srcs",
        "//src/java_tools/buildjar/javatests/com/google/devtools/build/java/bazel:srcs",
        "//src/java_tools/buildjar/javatests/com/google/devtools/build/java/turbine:srcs",
    ],
    visibility = ["//src:__pkg__"],
)

java_toolchain(
    name = "bootstrap_toolchain",
    bootclasspath = ["//tools/jdk:bootclasspath"],
    encoding = "UTF-8",
    extclasspath = ["//tools/jdk:extdir"],
    genclass = ["bootstrap_genclass_deploy.jar"],
    ijar = ["//third_party/ijar"],
    javabuilder = ["bootstrap_deploy.jar"],
    javac = ["//third_party/java/jdk/langtools:javac_jar"],
    jvm_opts = [
        "-XX:+TieredCompilation",
        "-XX:TieredStopAtLevel=1",
    ],
    singlejar = ["//src/java_tools/singlejar:bootstrap_deploy.jar"],
    source_version = "8",
    target_version = "8",
    visibility = ["//visibility:public"],
)

java_toolchain(
    name = "bootstrap_toolchain_jdk7",
    bootclasspath = ["//tools/jdk:bootclasspath"],
    encoding = "UTF-8",
    extclasspath = ["//tools/jdk:extdir"],
    genclass = ["bootstrap_genclass_deploy.jar"],
    ijar = ["//third_party/ijar"],
    javabuilder = ["//third_party/java/jdk/javabuilder:JavaBuilder_0.1.0"],
    javac = ["//third_party/java/jdk/langtools:javac7_jar"],
    jvm_opts = [
        "-XX:+TieredCompilation",
        "-XX:TieredStopAtLevel=1",
    ],
    singlejar = ["//src/java_tools/singlejar:bootstrap_deploy.jar"],
    source_version = "7",
    target_version = "7",
    visibility = ["//visibility:public"],
)
