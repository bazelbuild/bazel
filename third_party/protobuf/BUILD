package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(":proto_alias.bzl", "proto_alias")

PROTOBUF_VERSION = "3.0.0"

filegroup(
    name = "srcs",
    srcs = [
        "//third_party/protobuf/" + PROTOBUF_VERSION + ":srcs",
    ] + glob(["**"]),  # glob everything to satisfy the compile.sh srcs test
    visibility = ["//third_party:__pkg__"],
)

proto_alias("protoc", PROTOBUF_VERSION)

proto_alias("protobuf_java", PROTOBUF_VERSION)

proto_alias("protobuf-jars", PROTOBUF_VERSION)

proto_alias("protobuf-util", PROTOBUF_VERSION)

alias(
    name = "protobuf_java_util",
    actual = "//third_party/protobuf/3.0.0:protobuf-util"
)

proto_alias("protobuf_python", PROTOBUF_VERSION)

proto_alias("protobuf_clib", PROTOBUF_VERSION)

alias(
    name = "protobuf",
    actual = "//third_party/protobuf/3.0.0:protobuf_clib"
)

proto_alias("protoc_lib", PROTOBUF_VERSION)

proto_alias("java_toolchain", PROTOBUF_VERSION)
