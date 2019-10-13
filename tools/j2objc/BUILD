load("@rules_java//java:defs.bzl", "java_binary")
load("@rules_proto//proto:defs.bzl", "proto_lang_toolchain")

package(default_visibility = ["//visibility:public"])

exports_files(glob(["**"]))

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

java_binary(
    name = "j2objc",
    main_class = "com.google.devtools.j2objc.J2ObjC",
    runtime_deps = ["@bazel_j2objc//:j2objc"],
)

filegroup(
    name = "j2objc_wrapper",
    srcs = ["j2objc_wrapper.py"],
)

filegroup(
    name = "j2objc_header_map",
    srcs = ["j2objc_header_map.py"],
)

proto_lang_toolchain(
    name = "j2objc_proto_toolchain",
    blacklisted_protos = [],
    command_line = "--PLUGIN_j2objc_out=file_dir_mapping,generate_class_mappings:$(OUT)",
    plugin = "//third_party/java/j2objc:proto_plugin",
    runtime = "//third_party/java/j2objc:proto_runtime",
    visibility = ["//visibility:public"],
)
