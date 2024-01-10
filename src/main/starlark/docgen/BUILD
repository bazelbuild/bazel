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
)
