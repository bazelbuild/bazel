package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

# This rule is only available from Bazel 7.0.0
starlark_doc_extract(
    name = "gen_be_proto_stardoc_proto",
    src = "proto.bzl",
)

starlark_doc_extract(
    name = "gen_be_java_stardoc_proto",
    src = "java.bzl",
    deps = [
        "@protobuf//bazel/common:proto_info_bzl",
        "@rules_java//java/bazel/rules",
    ],
)

starlark_doc_extract(
    name = "gen_be_cpp_stardoc_proto",
    src = "cpp.bzl",
)

starlark_doc_extract(
    name = "gen_be_objc_stardoc_proto",
    src = "objc.bzl",
)

starlark_doc_extract(
    name = "gen_be_python_stardoc_proto",
    src = "python.bzl",
    deps = [
        "@rules_cc//cc/common",
        "@rules_cc//cc/toolchains:toolchain_rules",
        "@rules_python//python:py_binary_bzl",
        "@rules_python//python:py_library_bzl",
        "@rules_python//python:py_runtime_bzl",
        "@rules_python//python:py_test_bzl",
    ],
)
