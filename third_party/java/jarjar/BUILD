package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

java_import(
    name = "jarjar",
    jars = ["jarjar-1.4.jar"],
)

java_binary(
    name = "jarjar_bin",
    main_class = "com.tonicsystems.jarjar.Main",
    runtime_deps = [":jarjar"],
)
