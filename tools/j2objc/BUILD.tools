load("@rules_java//java:defs.bzl", "java_binary")
load("@rules_proto//proto:defs.bzl", "proto_lang_toolchain")

package(default_visibility = ["//visibility:public"])

exports_files(glob(["**"]))

java_binary(
    name = "j2objc",
    main_class = "com.google.devtools.j2objc.J2ObjC",
    runtime_deps = ["@bazel_j2objc//:j2objc"],
)

py_binary(
    name = "j2objc_wrapper_binary",
    srcs = ["j2objc_wrapper_binary.py"],
    python_version = "PY3",
)

py_binary(
    name = "j2objc_header_map_binary",
    srcs = ["j2objc_header_map_binary.py"],
    python_version = "PY3",
)

proto_lang_toolchain(
    name = "j2objc_proto_toolchain",
    blacklisted_protos = [],
    command_line = "--PLUGIN_j2objc_out=file_dir_mapping,generate_class_mappings:$(OUT)",
    plugin = "//third_party/java/j2objc:proto_plugin",
    plugin_format_flag = "--plugin=protoc-gen-PLUGIN_j2objc=%s",
    progress_message = "Generating j2objc proto_library %{label}",
    runtime = "//third_party/java/j2objc:proto_runtime",
    visibility = ["//visibility:public"],
)
