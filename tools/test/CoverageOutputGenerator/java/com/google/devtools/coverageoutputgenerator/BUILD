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
    srcs = ["BUILD.tools"],
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
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "BranchCoverage",
    srcs = ["BranchCoverage.java"],
    deps = [
        "//third_party:auto_value",
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
    ],
)

java_library(
    name = "MainLibrary",
    srcs = ["Main.java"],
    deps = [
        ":Constants",
        ":Coverage",
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
        ":all_lcov_merger_tools_deploy.jar",
    ],
    visibility = ["//visibility:public"],
)
