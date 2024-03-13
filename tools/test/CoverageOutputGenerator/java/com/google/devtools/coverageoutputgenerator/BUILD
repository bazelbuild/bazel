load("@rules_java//java:defs.bzl", "java_binary", "java_library")
load("//tools:build_defs.bzl", "transition_java_language_8_filegroup")

package(
    default_visibility = [
        "//tools/test/CoverageOutputGenerator/javatests/com/google/devtools/coverageoutputgenerator:__pkg__",
    ],
)

licenses(["notice"])  # Apache 2.0

java_binary(
    name = "all_lcov_merger_tools",
    visibility = ["//visibility:public"],
    runtime_deps = [":lcov_merger_lib"],
)

genrule(
    name = "rename_build_tools",
    srcs = ["BUILD.remote"],
    outs = ["coverage/BUILD"],
    cmd = "cp $< $@",
)

genrule(
    name = "create_empty_workspace",
    outs = ["WORKSPACE"],
    cmd = "touch $@",
)

genrule(
    name = "coverage_output_generator_zip",
    srcs = [
        ":all_lcov_merger_tools_deploy.jar",
        "coverage/BUILD",
        "WORKSPACE",
    ],
    outs = ["coverage_output_generator.zip"],
    cmd = "zip -q -j $@ $(SRCS)",
)

sh_binary(
    name = "build_and_upload",
    srcs = ["build_and_upload.sh"],
    data = [":coverage_output_generator.zip"],
)

java_library(
    name = "lcov_merger_lib",
    srcs = glob(["*.java"]),
    deps = [
        "//third_party:auto_value",
        "//third_party:gson",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/java/jcommander",
    ],
)

java_library(
    name = "BranchCoverage",
    srcs = ["BranchCoverage.java"],
    deps = [
        "//third_party:auto_value",
        "//third_party:guava",
    ],
)

java_library(
    name = "LineCoverage",
    srcs = ["LineCoverage.java"],
    deps = [
        "//third_party:auto_value",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "SourceFileCoverage",
    srcs = ["SourceFileCoverage.java"],
    deps = [
        ":BranchCoverage",
        ":LineCoverage",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "LcovPrinter",
    srcs = ["LcovPrinter.java"],
    deps = [
        ":BranchCoverage",
        ":Constants",
        ":Coverage",
        ":LineCoverage",
        ":SourceFileCoverage",
        "//third_party:guava",
    ],
)

java_library(
    name = "Constants",
    srcs = ["Constants.java"],
    deps = ["//third_party:guava"],
)

java_library(
    name = "GcovParser",
    srcs = [
        "GcovParser.java",
        "Parser.java",
    ],
    deps = [
        ":BranchCoverage",
        ":Constants",
        ":LineCoverage",
        ":SourceFileCoverage",
        "//third_party:guava",
    ],
)

java_library(
    name = "GcovJsonParser",
    srcs = [
        "GcovJsonParser.java",
        "Parser.java",
    ],
    deps = [
        ":BranchCoverage",
        ":Constants",
        ":LineCoverage",
        ":SourceFileCoverage",
        "//third_party:gson",
    ],
)

java_library(
    name = "LcovParser",
    srcs = [
        "LcovParser.java",
        "Parser.java",
    ],
    deps = [
        ":BranchCoverage",
        ":Constants",
        ":LineCoverage",
        ":SourceFileCoverage",
    ],
)

java_library(
    name = "Coverage",
    srcs = ["Coverage.java"],
    deps = [
        ":Constants",
        ":SourceFileCoverage",
        "//third_party:guava",
    ],
)

java_library(
    name = "LcovMergerFlags",
    srcs = ["LcovMergerFlags.java"],
    deps = [
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/java/jcommander",
    ],
)

java_library(
    name = "MainLibrary",
    srcs = ["Main.java"],
    deps = [
        ":Constants",
        ":Coverage",
        ":GcovJsonParser",
        ":GcovParser",
        ":LcovMergerFlags",
        ":LcovParser",
        ":LcovPrinter",
        ":SourceFileCoverage",
        "//third_party:guava",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "embedded_tools",
    srcs = [
        "BUILD.tools",
    ],
    visibility = ["//visibility:public"],
)

genrule(
    name = "coverage_lcov_merge_tools",
    srcs = [":all_lcov_merger_tools_deploy.jar"],
    outs = ["coverage/all_lcov_merger_tools_deploy.jar"],
    cmd = "cp $< $@",
)

genrule(
    name = "coverage_empty_workspace",
    outs = ["coverage/WORKSPACE"],
    cmd = "touch $@",
    visibility = ["//src/test/shell/bazel:__pkg__"],
)

# Used to act as an override for @remote_coverage_tools in integration tests
filegroup(
    name = "coverage_output_generator_repo",
    data = [
        ":coverage_empty_workspace",
        ":coverage_lcov_merge_tools",
        ":rename_build_tools",
    ],
    visibility = ["//src/test/shell/bazel:__pkg__"],
)

transition_java_language_8_filegroup(
    name = "coverage_output_generator_repo_java8",
    files = [
        ":coverage/BUILD",
        ":coverage/WORKSPACE",
        ":coverage/all_lcov_merger_tools_deploy.jar",
    ],
    visibility = ["//src/test/shell/bazel:__pkg__"],
)
