load("@rules_java//java:defs.bzl", "java_test")
load(":skydoc_test.bzl", "skydoc_test")

package(
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]) + [
        "//src/test/java/com/google/devtools/build/skydoc/rendering:srcs",
        "//src/test/java/com/google/devtools/build/skydoc/testdata/same_level_file_test:srcs",
    ],
    visibility = ["//src:__pkg__"],
)

java_test(
    name = "SkydocTest",
    size = "medium",
    srcs = ["SkydocTest.java"],
    visibility = ["//devtools/blaze/main:__pkg__"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages/semantics",
        "//src/main/java/com/google/devtools/build/skydoc:skydoc_lib",
        "//src/main/java/com/google/devtools/build/skydoc/fakebuildapi",
        "//src/main/java/com/google/devtools/build/skydoc/rendering",
        "//src/main/java/com/google/devtools/build/skydoc/rendering:rendering_util",
        "//src/main/java/com/google/devtools/build/skydoc/rendering/proto:stardoc_output_java_proto",
        "//src/main/java/net/starlark/java/eval",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestUtils",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
        "//tools/java/runfiles",
    ],
)

java_binary(
    name = "binaryprotoToTextproto",
    srcs = ["BinaryprotoToTextproto.java"],
    main_class = "com.google.devtools.build.skydoc.BinaryprotoToTextproto",
    visibility = ["//visibility:private"],
    deps = [
        "//src/main/java/com/google/devtools/build/skydoc/rendering/proto:stardoc_output_java_proto",
        "//third_party/protobuf:protobuf_java",
    ],
)

skydoc_test(
    name = "angle_bracket_test",
    golden_file = "testdata/angle_bracket_test/golden.textproto",
    input_file = "testdata/angle_bracket_test/input.bzl",
)

skydoc_test(
    name = "simple_test",
    golden_file = "testdata/simple_test/golden.textproto",
    input_file = "testdata/simple_test/input.bzl",
    symbol_names = ["my_rule"],
)

skydoc_test(
    name = "repo_rule_test",
    golden_file = "testdata/repo_rules_test/golden.textproto",
    input_file = "testdata/repo_rules_test/input.bzl",
)

skydoc_test(
    name = "unknown_name",
    golden_file = "testdata/unknown_name_test/golden.textproto",
    input_file = "testdata/unknown_name_test/input.bzl",
)

skydoc_test(
    name = "multiple_rules_test",
    golden_file = "testdata/multiple_rules_test/golden.textproto",
    input_file = "testdata/multiple_rules_test/input.bzl",
)

skydoc_test(
    name = "unknown_predefines_test",
    golden_file = "testdata/unknown_predefines_test/golden.textproto",
    input_file = "testdata/unknown_predefines_test/input.bzl",
    symbol_names = ["my_rule"],
)

skydoc_test(
    name = "apple_basic_test",
    golden_file = "testdata/apple_basic_test/golden.textproto",
    input_file = "testdata/apple_basic_test/input.bzl",
    symbol_names = ["apple_related_rule"],
)

skydoc_test(
    name = "multiple_files_test",
    golden_file = "testdata/multiple_files_test/golden.textproto",
    input_file = "testdata/multiple_files_test/input.bzl",
    deps = [
        "testdata/multiple_files_test/dep.bzl",
        "testdata/multiple_files_test/inner_dep.bzl",
    ],
)

skydoc_test(
    name = "same_level_file_test",
    golden_file = "//src/test/java/com/google/devtools/build/skydoc/testdata/same_level_file_test:golden.textproto",
    input_file = "//src/test/java/com/google/devtools/build/skydoc/testdata/same_level_file_test:input.bzl",
    symbol_names = ["my_rule"],
    deps = [
        "//src/test/java/com/google/devtools/build/skydoc/testdata/same_level_file_test:dep.bzl",
    ],
)

skydoc_test(
    name = "misc_apis_test",
    golden_file = "testdata/misc_apis_test/golden.textproto",
    input_file = "testdata/misc_apis_test/input.bzl",
)

skydoc_test(
    name = "attribute_types_test",
    golden_file = "testdata/attribute_types_test/golden.textproto",
    input_file = "testdata/attribute_types_test/input.bzl",
    symbol_names = ["my_rule"],
)

skydoc_test(
    name = "filter_rules_test",
    golden_file = "testdata/filter_rules_test/golden.textproto",
    input_file = "testdata/filter_rules_test/input.bzl",
    symbol_names = [
        "my_rule",
        "allowlisted_dep_rule",
    ],
    deps = [
        "testdata/filter_rules_test/dep.bzl",
    ],
)

skydoc_test(
    name = "provider_basic_test",
    golden_file = "testdata/provider_basic_test/golden.textproto",
    input_file = "testdata/provider_basic_test/input.bzl",
)

skydoc_test(
    name = "function_basic_test",
    golden_file = "testdata/function_basic_test/golden.textproto",
    input_file = "testdata/function_basic_test/input.bzl",
)

skydoc_test(
    name = "function_wrap_multiple_lines_test",
    golden_file = "testdata/function_wrap_multiple_lines_test/golden.textproto",
    input_file = "testdata/function_wrap_multiple_lines_test/input.bzl",
)

skydoc_test(
    name = "namespace_test",
    golden_file = "testdata/namespace_test/golden.textproto",
    input_file = "testdata/namespace_test/input.bzl",
)

skydoc_test(
    name = "namespace_test_with_allowlist",
    golden_file = "testdata/namespace_test/golden.textproto",
    input_file = "testdata/namespace_test/input.bzl",
    symbol_names = [
        "my_namespace",
    ],
)

skydoc_test(
    name = "multi_level_namespace_test",
    golden_file = "testdata/multi_level_namespace_test/golden.textproto",
    input_file = "testdata/multi_level_namespace_test/input.bzl",
)

skydoc_test(
    name = "multi_level_namespace_test_with_allowlist",
    golden_file = "testdata/multi_level_namespace_test_with_allowlist/golden.textproto",
    input_file = "testdata/multi_level_namespace_test_with_allowlist/input.bzl",
    symbol_names = [
        "my_namespace",
        "other_namespace.foo.nothing",
    ],
)

skydoc_test(
    name = "macro_kwargs_test",
    golden_file = "testdata/macro_kwargs_test/golden.textproto",
    input_file = "testdata/macro_kwargs_test/input.bzl",
)

skydoc_test(
    name = "struct_default_value_test",
    golden_file = "testdata/struct_default_value_test/golden.textproto",
    input_file = "testdata/struct_default_value_test/input.bzl",
)

skydoc_test(
    name = "aspect_test",
    golden_file = "testdata/aspect_test/golden.textproto",
    input_file = "testdata/aspect_test/input.bzl",
)

skydoc_test(
    name = "providers_for_attributes_test",
    golden_file = "testdata/providers_for_attributes_test/golden.textproto",
    input_file = "testdata/providers_for_attributes_test/input.bzl",
    deps = [
        "testdata/providers_for_attributes_test/dep.bzl",
    ],
)

skydoc_test(
    name = "pure_markdown_template_test",
    format = "markdown",
    golden_file = "testdata/pure_markdown_template_test/golden.textproto",
    input_file = "testdata/pure_markdown_template_test/input.bzl",
)

skydoc_test(
    name = "attribute_defaults_test",
    format = "markdown",
    golden_file = "testdata/attribute_defaults_test/golden.textproto",
    input_file = "testdata/attribute_defaults_test/input.bzl",
)

genrule(
    name = "generate_bzl_test_dep",
    srcs = ["testdata/generated_bzl_test/dep.bzl.tpl"],
    outs = ["testdata/generated_bzl_test/dep.bzl"],
    cmd = "cp $< $@",
)

skydoc_test(
    name = "generated_bzl_test",
    golden_file = "testdata/generated_bzl_test/golden.textproto",
    input_file = "testdata/generated_bzl_test/input.bzl",
    deps = [
        "testdata/generated_bzl_test/dep.bzl",
    ],
)
