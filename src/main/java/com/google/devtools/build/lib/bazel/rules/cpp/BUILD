load("//:distdir_deps.bzl", "gen_workspace_stanza")
load("@rules_java//java:defs.bzl", "java_library")

package(
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(
        ["*"],
        # The file is currently real. On the PR that deletes it, it seems to be
        # in the distfile, so integration tests fail because we are including
        # # cc_configure.WORKSPACE twice - once in the glob and once explicitly.
        # Excluding it here allows the PR to work.
        # TODO(#12081): Remove this exclude after #12743 is merged.
        exclude = ["cc_configure.WORKSPACE"],
    ) + [
        # We need the generated file in srcs so it gets in the source distfile.
        # That allows the boostrap build to work.
        ":cc_configure.WORKSPACE",
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
        "//src/main/java/com/google/devtools/build/lib/analysis:config/host_transition",
        "//src/main/java/com/google/devtools/build/lib/analysis:platform_configuration",
        "//src/main/java/com/google/devtools/build/lib/analysis:rule_definition_environment",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/cpp",
        "//src/main/java/com/google/devtools/build/lib/util:filetype",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/java/net/starlark/java/syntax",
        "//third_party:guava",
    ],
)

java_library(
    name = "bazel_cpp_semantics",
    srcs = ["BazelCppSemantics.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/build_configuration",
        "//src/main/java/com/google/devtools/build/lib/analysis:transitive_info_collection",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//third_party:guava",
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
)
