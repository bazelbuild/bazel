java_binary(
    name = "GenClass",
    main_class = "com.google.devtools.build.buildjar.genclass.GenClass",
    visibility = ["//visibility:public"],
    runtime_deps = [":genclass_lib"],
)

java_library(
    name = "genclass_lib",
    srcs = glob(["*.java"]),
    deps = [
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/jarhelper",
        "//src/main/protobuf:proto_java_compilation",
        "//third_party:guava",
    ],
)
