package(default_visibility = ["//src:__subpackages__"])

java_binary(
    name = "JavaBuilder",
    main_class = "com.google.devtools.build.buildjar.BazelJavaBuilder",
    visibility = ["//visibility:public"],
    runtime_deps = ["//src/java_tools/buildjar/java/com/google/devtools/build/buildjar"],
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
