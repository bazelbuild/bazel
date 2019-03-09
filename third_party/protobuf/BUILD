package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(":proto_alias.bzl", "proto_alias")

PROTOBUF_VERSION = "3.6.1"
_NEW_PROTOBUF_VERSION = "3.7.1"

filegroup(
    name = "srcs",
    srcs = [
        "//third_party/protobuf/" + PROTOBUF_VERSION + ":srcs",
        "//third_party/protobuf/" + _NEW_PROTOBUF_VERSION + ":srcs",
    ] + glob(["**"]),  # glob everything to satisfy the compile.sh srcs test
    visibility = ["//third_party:__pkg__"],
)

proto_alias("protoc", PROTOBUF_VERSION)

proto_alias("protobuf_java", PROTOBUF_VERSION)

proto_alias("protobuf-jars", PROTOBUF_VERSION)

proto_alias("protobuf_java_util", PROTOBUF_VERSION)

proto_alias("protobuf_python", PROTOBUF_VERSION)

proto_alias("protobuf", PROTOBUF_VERSION)

proto_alias("protobuf_headers", PROTOBUF_VERSION)

proto_alias("protoc_lib", PROTOBUF_VERSION)

proto_alias("java_toolchain", PROTOBUF_VERSION)
