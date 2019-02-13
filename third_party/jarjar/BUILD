package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

# jarjar_bin
java_binary(
    name = "jarjar_command",
    srcs = glob(
        ["java/**/*.java"],
        exclude = [
            "java/com/tonicsystems/jarjar/util/AntJarProcessor.java",
            "java/com/tonicsystems/jarjar/JarJarMojo.java",
            "java/com/tonicsystems/jarjar/JarJarTask.java",
        ],
    ),
    main_class = "com.tonicsystems.jarjar.Main",
    resources = ["java/com/tonicsystems/jarjar/help.txt"],
    deps = [
        "//third_party:asm",
        "//third_party:asm-commons",
        "//third_party:asm-util",
        "//third_party:jsr305",
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
