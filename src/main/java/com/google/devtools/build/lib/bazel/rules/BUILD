load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["*"]) + [
        "//src/main/java/com/google/devtools/build/lib/bazel/rules/cpp:srcs",
        "//src/main/java/com/google/devtools/build/lib/bazel/rules/genrule:srcs",
        "//src/main/java/com/google/devtools/build/lib/bazel/rules/java:srcs",
        "//src/main/java/com/google/devtools/build/lib/bazel/rules/python:srcs",
    ],
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "rules",
    srcs = glob(["*.java"]),
    resources = [
        ":builtins_bzl_zip",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/analysis:actions/file_write_action_context",
        "//src/main/java/com/google/devtools/build/lib/analysis:actions/template_expansion_action",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/analysis:common_prerequisite_validator",
        "//src/main/java/com/google/devtools/build/lib/analysis:constraints/environment_rule",
        "//src/main/java/com/google/devtools/build/lib/analysis:platform_configuration",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_configuration",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:core_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:fragment",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:fragment_options",
        "//src/main/java/com/google/devtools/build/lib/bazel:bazel_configuration",
        "//src/main/java/com/google/devtools/build/lib/bazel/rules/cpp",
        "//src/main/java/com/google/devtools/build/lib/bazel/rules/genrule",
        "//src/main/java/com/google/devtools/build/lib/bazel/rules/java:bazel_java_semantics",
        "//src/main/java/com/google/devtools/build/lib/bazel/rules/python",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/exec:execution_options",
        "//src/main/java/com/google/devtools/build/lib/exec:module_action_context_registry",
        "//src/main/java/com/google/devtools/build/lib/exec:spawn_cache",
        "//src/main/java/com/google/devtools/build/lib/exec:spawn_strategy_registry",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/packages/semantics",
        "//src/main/java/com/google/devtools/build/lib/remote",
        "//src/main/java/com/google/devtools/build/lib/remote/options",
        "//src/main/java/com/google/devtools/build/lib/rules:alias",
        "//src/main/java/com/google/devtools/build/lib/rules:extra/action_listener_rule",
        "//src/main/java/com/google/devtools/build/lib/rules:extra/extra_action_rule",
        "//src/main/java/com/google/devtools/build/lib/rules:filegroup/filegroup",
        "//src/main/java/com/google/devtools/build/lib/rules:label_build_settings",
        "//src/main/java/com/google/devtools/build/lib/rules:toolchain_type",
        "//src/main/java/com/google/devtools/build/lib/rules/android",
        "//src/main/java/com/google/devtools/build/lib/rules/apple",
        "//src/main/java/com/google/devtools/build/lib/rules/config",
        "//src/main/java/com/google/devtools/build/lib/rules/core",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp",
        "//src/main/java/com/google/devtools/build/lib/rules/genquery",
        "//src/main/java/com/google/devtools/build/lib/rules/genrule",
        "//src/main/java/com/google/devtools/build/lib/rules/java:java-compilation",
        "//src/main/java/com/google/devtools/build/lib/rules/objc",
        "//src/main/java/com/google/devtools/build/lib/rules/platform",
        "//src/main/java/com/google/devtools/build/lib/rules/proto",
        "//src/main/java/com/google/devtools/build/lib/rules/python",
        "//src/main/java/com/google/devtools/build/lib/rules/starlarkdocextract",
        "//src/main/java/com/google/devtools/build/lib/rules/test:test_suite_rule",
        "//src/main/java/com/google/devtools/build/lib/rules/test:testing_support_rules",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/core",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/cpp",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/objc",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//src/main/java/com/google/devtools/build/lib/util:string_encoding",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/net/starlark/java/eval",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

# We need this redirect target so that the builtins zip can be packaged as a
# resource next to the BazelBuiltins class (com/google/devtools/...). Otherwise
# it gets placed under the package path of the original zip target.
genrule(
    name = "builtins_bzl_zip",
    srcs = ["//src/main/starlark/builtins_bzl:builtins_bzl.zip"],
    outs = ["builtins_bzl.zip"],
    cmd = "cp $(SRCS) $@",
    visibility = ["//:__pkg__"],
)
