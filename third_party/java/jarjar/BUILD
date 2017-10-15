package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

filegroup(
    name = "embedded_tools_srcs",
    srcs = [
        "BUILD.tools",
        "LICENSE",
        ":jarjar_command_deploy.jar",
    ],
)

java_library(
    name = "jarjar_core",
    srcs = glob(["jarjar-core/src/main/**/*.java"]),
    deps = [
        "//third_party:asm",
        "//third_party:asm-commons",
        "//third_party:asm-util",
        "//third_party:jsr305",
        "//third_party:slf4j",
    ],
)

java_binary(
    name = "jarjar_command",
    srcs = glob(["jarjar-command/src/main/**/*.java"]),
    main_class = "com.tonicsystems.jarjar.Main",
    deps = [
        ":jarjar_core",
        "//third_party:jsr305",
        "//third_party/java/jopt-simple",
    ],
)

java_import(
    name = "jarjar_import",
    jars = [":jarjar_command_deploy.jar"],
)

java_binary(
    name = "jarjar_bin",
    main_class = "com.tonicsystems.jarjar.Main",
    runtime_deps = [":jarjar_import"],
)
