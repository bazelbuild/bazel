load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["*"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "info",
    srcs = glob(["*.java"]),
    resources = [
        ":gen_rule_src_doc_proto",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/docgen:visible_to_info",
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib:runtime/memory_pressure",
        "//src/main/java/com/google/devtools/build/lib/actions:localhost_capacity",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/analysis:blaze_version_info",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/build_configuration",
        "//src/main/java/com/google/devtools/build/lib/bazel/rules",
        "//src/main/java/com/google/devtools/build/lib/bugreport",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/packages:rule_class_utils",
        "//src/main/java/com/google/devtools/build/lib/packages/semantics",
        "//src/main/java/com/google/devtools/build/lib/pkgcache",
        "//src/main/java/com/google/devtools/build/lib/skyframe:skyframe_cluster",
        "//src/main/java/com/google/devtools/build/lib/skyframe:starlark_builtins_value",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:abrupt_exit_exception",
        "//src/main/java/com/google/devtools/build/lib/util:debug-logger-configurator",
        "//src/main/java/com/google/devtools/build/lib/util:detailed_exit_code",
        "//src/main/java/com/google/devtools/build/lib/util:heap_offset_helper",
        "//src/main/java/com/google/devtools/build/lib/util:string",
        "//src/main/java/com/google/devtools/build/lib/util:string_encoding",
        "//src/main/java/com/google/devtools/build/lib/util/io:out-err",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/worker:worker_process_metrics",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/net/starlark/java/annot",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/protobuf:build_java_proto",
        "//src/main/protobuf:builtin_java_proto",
        "//src/main/protobuf:failure_details_java_proto",
        "//src/main/protobuf:rule_src_doc_java_proto",
        "//third_party:flogger",
        "//third_party:guava",
    ],
)

genrule(
    name = "gen_rule_src_doc_proto",
    srcs = [
        "//src/main/java/com/google/devtools/build/docgen:bazel_link_map",
        "//src/main/starlark/docgen:gen_be_proto_stardoc_proto",
        "//src/main/starlark/docgen:gen_be_java_stardoc_proto",
        "//src/main/starlark/docgen:gen_be_cpp_stardoc_proto",
        "//src/main/starlark/docgen:gen_be_objc_stardoc_proto",
        "//src/main/starlark/docgen:gen_be_python_stardoc_proto",
        "//src/main/starlark/docgen:gen_be_shell_stardoc_proto",
        "//src/main/java/com/google/devtools/build/lib:docs_embedded_in_sources",
    ],
    outs = ["rule_src_doc.pb"],
    cmd = (
        "$(location //src/main/java/com/google/devtools/build/docgen:rule_src_doc_exporter)" +
        " --output_file=$@" +
        " --link_map_path=$(location //src/main/java/com/google/devtools/build/docgen:bazel_link_map) " +
        " --provider=com.google.devtools.build.lib.bazel.rules.BazelRuleClassProvider" +
        " --input_root=$$PWD" +
        " --input_dir=$$PWD/src/main/java/com/google/devtools/build/lib" +
        " --input_stardoc_proto=$(location //src/main/starlark/docgen:gen_be_proto_stardoc_proto)" +
        " --input_stardoc_proto=$(location //src/main/starlark/docgen:gen_be_java_stardoc_proto)" +
        " --input_stardoc_proto=$(location //src/main/starlark/docgen:gen_be_cpp_stardoc_proto)" +
        " --input_stardoc_proto=$(location //src/main/starlark/docgen:gen_be_objc_stardoc_proto)" +
        " --input_stardoc_proto=$(location //src/main/starlark/docgen:gen_be_python_stardoc_proto)" +
        " --input_stardoc_proto=$(location //src/main/starlark/docgen:gen_be_shell_stardoc_proto)"
    ),
    tools = [
        "//src/main/java/com/google/devtools/build/docgen:rule_src_doc_exporter",
    ],
)
