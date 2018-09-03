java_import(
    name = "protobuf",
    jars = ["libprotobuf_java.jar"],
)

proto_lang_toolchain(
    name = "java_toolchain",
    command_line = "--java_out=shared,immutable:$(OUT)",
    runtime = ":protobuf",
    visibility = ["//visibility:public"],
)

