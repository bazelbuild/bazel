package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

java_import(
    name = "jarjar_import",
    jars = [":jarjar_command_deploy.jar"],
)

java_binary(
    name = "jarjar_bin",
    main_class = "com.tonicsystems.jarjar.Main",
    runtime_deps = [":jarjar_import"],
)
