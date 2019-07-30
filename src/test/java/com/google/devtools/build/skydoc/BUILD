load(":skydoc_test.bzl", "skydoc_test")

package(
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]) + [
        "//src/test/java/com/google/devtools/build/skydoc/testdata/same_level_file_test:srcs",
    ],
    visibility = ["//src:__pkg__"],
)

filegroup(
    name = "test_template_files",
    srcs = [
        "//src/test/java/com/google/devtools/build/skydoc:test_templates/aspect.vm",
        "//src/test/java/com/google/devtools/build/skydoc:test_templates/func.vm",
        "//src/test/java/com/google/devtools/build/skydoc:test_templates/header.vm",
        "//src/test/java/com/google/devtools/build/skydoc:test_templates/provider.vm",
        "//src/test/java/com/google/devtools/build/skydoc:test_templates/rule.vm",
    ],
    tags = ["bazel"],
)

java_test(
    name = "SkydocTest",
    size = "medium",
    srcs = ["SkydocTest.java"],
    visibility = ["//devtools/blaze/main:__pkg__"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:skylark_semantics",
        "//src/main/java/com/google/devtools/build/lib:syntax",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/skydoc:skydoc_lib",
        "//src/main/java/com/google/devtools/build/skydoc/fakebuildapi",
        "//src/main/java/com/google/devtools/build/skydoc/rendering",
        "//src/main/java/com/google/devtools/build/skydoc/rendering/proto:stardoc_output_java_proto",
        "//src/test/java/com/google/devtools/build/lib:testutil",
        "//src/test/java/com/google/devtools/build/lib/skylark:testutil",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "MarkdownRendererTest",
    srcs = ["MarkdownRendererTest.java"],
    resources = [
        ":test_template_files",
    ],
    visibility = ["//devtools/blaze/main:__pkg__"],
    deps = [
        "//src/main/java/com/google/devtools/build/skydoc/rendering",
        "//src/main/java/com/google/devtools/build/skydoc/rendering/proto:stardoc_output_java_proto",
        "//src/test/java/com/google/devtools/build/lib:testutil",
        "//src/test/java/com/google/devtools/build/lib/skylark:testutil",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

skydoc_test(
    name = "input_template_test",
    func_template = "testdata/input_template_test/func.vm",
    golden_file = "testdata/input_template_test/golden.txt",
    header_template = "testdata/input_template_test/header.vm",
    input_file = "testdata/input_template_test/input.bzl",
    provider_template = "testdata/input_template_test/provider.vm",
    rule_template = "testdata/input_template_test/rule.vm",
)

skydoc_test(
    name = "angle_bracket_test",
    golden_file = "testdata/angle_bracket_test/golden.txt",
    input_file = "testdata/angle_bracket_test/input.bzl",
)

skydoc_test(
    name = "proto_format_test",
    format = "proto",
    golden_file = "testdata/proto_format_test/golden.raw",
    input_file = "testdata/proto_format_test/input.bzl",
)

skydoc_test(
    name = "cc_api_test",
    golden_file = "testdata/cc_api_test/golden.txt",
    input_file = "testdata/cc_api_test/input.bzl",
)

skydoc_test(
    name = "simple_test",
    golden_file = "testdata/simple_test/golden.txt",
    input_file = "testdata/simple_test/input.bzl",
    whitelisted_symbols = ["my_rule"],
)

skydoc_test(
    name = "repo_rule_test",
    golden_file = "testdata/repo_rules_test/golden.txt",
    input_file = "testdata/repo_rules_test/input.bzl",
)

skydoc_test(
    name = "unknown_name",
    golden_file = "testdata/unknown_name_test/golden.txt",
    input_file = "testdata/unknown_name_test/input.bzl",
)

skydoc_test(
    name = "multiple_rules_test",
    golden_file = "testdata/multiple_rules_test/golden.txt",
    input_file = "testdata/multiple_rules_test/input.bzl",
)

skydoc_test(
    name = "android_basic_test",
    golden_file = "testdata/android_basic_test/golden.txt",
    input_file = "testdata/android_basic_test/input.bzl",
    semantic_flags = [
        "--experimental_google_legacy_api",
    ],
    whitelisted_symbols = ["android_related_rule"],
)

skydoc_test(
    name = "apple_basic_test",
    golden_file = "testdata/apple_basic_test/golden.txt",
    input_file = "testdata/apple_basic_test/input.bzl",
    whitelisted_symbols = ["apple_related_rule"],
)

skydoc_test(
    name = "cpp_basic_test",
    golden_file = "testdata/cpp_basic_test/golden.txt",
    input_file = "testdata/cpp_basic_test/input.bzl",
    whitelisted_symbols = ["cpp_related_rule"],
)

skydoc_test(
    name = "java_basic_test",
    golden_file = "testdata/java_basic_test/golden.txt",
    input_file = "testdata/java_basic_test/input.bzl",
    whitelisted_symbols = ["java_related_rule"],
)

skydoc_test(
    name = "multiple_files_test",
    golden_file = "testdata/multiple_files_test/golden.txt",
    input_file = "testdata/multiple_files_test/input.bzl",
    deps = [
        "testdata/multiple_files_test/dep.bzl",
        "testdata/multiple_files_test/inner_dep.bzl",
    ],
)

skydoc_test(
    name = "same_level_file_test",
    golden_file = "//src/test/java/com/google/devtools/build/skydoc/testdata/same_level_file_test:golden.txt",
    input_file = "//src/test/java/com/google/devtools/build/skydoc/testdata/same_level_file_test:input.bzl",
    whitelisted_symbols = ["my_rule"],
    deps = [
        "//src/test/java/com/google/devtools/build/skydoc/testdata/same_level_file_test:dep.bzl",
    ],
)

skydoc_test(
    name = "misc_apis_test",
    golden_file = "testdata/misc_apis_test/golden.txt",
    input_file = "testdata/misc_apis_test/input.bzl",
)

skydoc_test(
    name = "attribute_types_test",
    golden_file = "testdata/attribute_types_test/golden.txt",
    input_file = "testdata/attribute_types_test/input.bzl",
    whitelisted_symbols = ["my_rule"],
)

skydoc_test(
    name = "filter_rules_test",
    golden_file = "testdata/filter_rules_test/golden.txt",
    input_file = "testdata/filter_rules_test/input.bzl",
    whitelisted_symbols = [
        "my_rule",
        "whitelisted_dep_rule",
    ],
    deps = [
        "testdata/filter_rules_test/dep.bzl",
    ],
)

skydoc_test(
    name = "provider_basic_test",
    golden_file = "testdata/provider_basic_test/golden.txt",
    input_file = "testdata/provider_basic_test/input.bzl",
)

skydoc_test(
    name = "function_basic_test",
    golden_file = "testdata/function_basic_test/golden.txt",
    input_file = "testdata/function_basic_test/input.bzl",
)

skydoc_test(
    name = "namespace_test",
    golden_file = "testdata/namespace_test/golden.txt",
    input_file = "testdata/namespace_test/input.bzl",
)

skydoc_test(
    name = "namespace_test_with_whitelist",
    golden_file = "testdata/namespace_test/golden.txt",
    input_file = "testdata/namespace_test/input.bzl",
    whitelisted_symbols = [
        "my_namespace",
    ],
)

skydoc_test(
    name = "multi_level_namespace_test",
    golden_file = "testdata/multi_level_namespace_test/golden.txt",
    input_file = "testdata/multi_level_namespace_test/input.bzl",
)

skydoc_test(
    name = "multi_level_namespace_test_with_whitelist",
    golden_file = "testdata/multi_level_namespace_test_with_whitelist/golden.txt",
    input_file = "testdata/multi_level_namespace_test_with_whitelist/input.bzl",
    whitelisted_symbols = [
        "my_namespace",
        "other_namespace.foo.nothing",
    ],
)

skydoc_test(
    name = "macro_kwargs_test",
    golden_file = "testdata/macro_kwargs_test/golden.txt",
    input_file = "testdata/macro_kwargs_test/input.bzl",
)

skydoc_test(
    name = "py_rule_test",
    golden_file = "testdata/py_rule_test/golden.txt",
    input_file = "testdata/py_rule_test/input.bzl",
    whitelisted_symbols = ["py_related_rule"],
)

skydoc_test(
    name = "struct_default_value_test",
    golden_file = "testdata/struct_default_value_test/golden.txt",
    input_file = "testdata/struct_default_value_test/input.bzl",
)

skydoc_test(
    name = "aspect_test",
    golden_file = "testdata/aspect_test/golden.txt",
    input_file = "testdata/aspect_test/input.bzl",
)

genrule(
    name = "generate_bzl_test_dep",
    srcs = ["testdata/generated_bzl_test/dep.bzl.tpl"],
    outs = ["testdata/generated_bzl_test/dep.bzl"],
    cmd = "cp $< $@",
)

skydoc_test(
    name = "generated_bzl_test",
    golden_file = "testdata/generated_bzl_test/golden.txt",
    input_file = "testdata/generated_bzl_test/input.bzl",
    deps = [
        "testdata/generated_bzl_test/dep.bzl",
    ],
)
