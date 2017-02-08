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

proto_alias("protobuf", PROTOBUF_VERSION)

# "protobuf" above refers to the Java proto runtime, but upstream, 
# "protobuf" refers to the C++ proto runtime. This alias allows us
# to switch all Bazel refernces to "protobuf" to the correct "protobuf_java".
# This alias should be temporary while this transition is done.
alias(
    name = "protobuf_java",
    actual = "//third_party/protobuf/3.0.0:protobuf"
)

proto_alias("protobuf-jars", PROTOBUF_VERSION)

proto_alias("protobuf-util", PROTOBUF_VERSION)

proto_alias("protobuf_python_srcs", PROTOBUF_VERSION)

proto_alias("protobuf_python", PROTOBUF_VERSION)

proto_alias("protobuf_lite", PROTOBUF_VERSION)

proto_alias("protobuf_clib", PROTOBUF_VERSION)

proto_alias("protoc_lib", PROTOBUF_VERSION)

proto_alias("java_toolchain", PROTOBUF_VERSION)
