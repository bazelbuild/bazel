"""Starlark rules for building Protocol Buffers."""

load("//tools/proto/private/rules:proto_toolchain.bzl", _proto_toolchain = "proto_toolchain")

# Rules.
proto_toolchain = _proto_toolchain
