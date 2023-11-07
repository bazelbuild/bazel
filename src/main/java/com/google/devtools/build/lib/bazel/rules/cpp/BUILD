load("@rules_java//java:defs.bzl", "java_library")
load("//:workspace_deps.bzl", "gen_workspace_stanza")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["*"]) + [
        "//src/main/java/com/google/devtools/build/lib/bazel/rules/cpp/proto:srcs",
    ],
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "cpp",
    srcs = glob(
        ["*.java"],
        exclude = ["BazelCppSemantics.java"],
    ),
    resources = [":cc_configure.WORKSPACE"],
    deps = [
        ":bazel_cpp_semantics",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/execution_transition_factory",
        "//src/main/java/com/google/devtools/build/lib/analysis:platform_configuration",
        "//src/main/java/com/google/devtools/build/lib/analysis:rule_definition_environment",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/cpp",
        "//src/main/java/com/google/devtools/build/lib/util:filetype",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "bazel_cpp_semantics",
    srcs = ["BazelCppSemantics.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/build_configuration",
        "//src/main/java/com/google/devtools/build/lib/analysis:rule_error_consumer",
        "//src/main/java/com/google/devtools/build/lib/analysis:transitive_info_collection",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec:serialization-constant",
        "//src/main/java/net/starlark/java/eval",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

gen_workspace_stanza(
    name = "workspace_with_rules_cc",
    out = "cc_configure.WORKSPACE",
    postamble = """
        cc_configure()
    """,
    preamble = """
        load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
        load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
        load("@bazel_tools//tools/cpp:cc_configure.bzl", "cc_configure")
    """,
    repos = ["rules_cc"],
    use_maybe = 1,
    visibility = ["//:__pkg__"],
)
