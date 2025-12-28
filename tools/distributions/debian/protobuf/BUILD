load("@com_google_protobuf//bazel/toolchains:proto_lang_toolchain.bzl", "proto_lang_toolchain")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

# Protobuf targets needed for building //src:bazel_nojdk
# @protobuf//:protobuf_headers
# @protobuf//:protobuf_java
# @protobuf//:protobuf_java_util
# @protobuf//:protoc
# @protobuf//:protoc_lib
# @protobuf//:any_proto
# @protobuf//:descriptor_proto
# @protobuf//:duration_proto
# @protobuf//:empty_proto
# @protobuf//:timestamp_proto
# @protobuf//:wrappers_proto
# @protobuf//:cc_toolchain
# @protobuf//:java_toolchain

# This could be empty because all headers are installed at /usr/include
# which is the default search path for gcc on Debian.
cc_library(
    name = "protobuf_headers",
)

alias(
    name = "protobuf_java",
    actual = "@debian_java_deps//:protobuf_java",
    tags = ["manual"],
)

alias(
    name = "protobuf_java_util",
    actual = "@debian_java_deps//:protobuf_java_util",
    tags = ["manual"],
)

alias(
    name = "protoc",
    actual = "@debian_bin_deps//:protobuf-compiler",
    tags = ["manual"],
)

alias(
    name = "protoc_lib",
    actual = "@debian_cc_deps//:protoc_lib",
    tags = ["manual"],
)

alias(
    name = "any_proto",
    actual = "@debian_proto_deps//:any_proto",
    tags = ["manual"],
)

alias(
    name = "descriptor_proto",
    actual = "@debian_proto_deps//:descriptor_proto",
    tags = ["manual"],
)

alias(
    name = "duration_proto",
    actual = "@debian_proto_deps//:duration_proto",
    tags = ["manual"],
)

alias(
    name = "empty_proto",
    actual = "@debian_proto_deps//:empty_proto",
    tags = ["manual"],
)

alias(
    name = "timestamp_proto",
    actual = "@debian_proto_deps//:timestamp_proto",
    tags = ["manual"],
)

alias(
    name = "wrappers_proto",
    actual = "@debian_proto_deps//:wrappers_proto",
    tags = ["manual"],
)

proto_lang_toolchain(
    name = "cc_toolchain",
    command_line = "--cpp_out=$(OUT)",
    runtime = ":protoc_lib",
    tags = ["manual"],
)

proto_lang_toolchain(
    name = "java_toolchain",
    command_line = "--java_out=$(OUT)",
    runtime = ":protobuf_java",
    tags = ["manual"],
)
