load("@rules_java//java:java_library.bzl", "java_library")
load("@rules_java//java:java_test.bzl", "java_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

java_test(
    name = "BranchCoverageTest",
    srcs = ["BranchCoverageTest.java"],
    deps = [
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
        "//tools/test/CoverageOutputGenerator/java/com/google/devtools/coverageoutputgenerator:BranchCoverage",
    ],
)

java_test(
    name = "LineCoverageTest",
    srcs = ["LineCoverageTest.java"],
    deps = [
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
        "//tools/test/CoverageOutputGenerator/java/com/google/devtools/coverageoutputgenerator:LineCoverage",
    ],
)

java_test(
    name = "SourceFileCoverageTest",
    srcs = ["SourceFileCoverageTest.java"],
    deps = [
        ":LineCoverageTest",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
        "//tools/test/CoverageOutputGenerator/java/com/google/devtools/coverageoutputgenerator:BranchCoverage",
        "//tools/test/CoverageOutputGenerator/java/com/google/devtools/coverageoutputgenerator:LineCoverage",
        "//tools/test/CoverageOutputGenerator/java/com/google/devtools/coverageoutputgenerator:SourceFileCoverage",
    ],
)

java_test(
    name = "LcovPrinterTest",
    srcs = ["LcovPrinterTest.java"],
    tags = [
        # TODO(laszlocsomor): fix on Windows or describe why it cannot pass.
        "no_windows",
    ],
    deps = [
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
        "//tools/test/CoverageOutputGenerator/java/com/google/devtools/coverageoutputgenerator:BranchCoverage",
        "//tools/test/CoverageOutputGenerator/java/com/google/devtools/coverageoutputgenerator:Constants",
        "//tools/test/CoverageOutputGenerator/java/com/google/devtools/coverageoutputgenerator:Coverage",
        "//tools/test/CoverageOutputGenerator/java/com/google/devtools/coverageoutputgenerator:LcovPrinter",
        "//tools/test/CoverageOutputGenerator/java/com/google/devtools/coverageoutputgenerator:SourceFileCoverage",
    ],
)

java_test(
    name = "LcovParserTest",
    srcs = ["LcovParserTest.java"],
    deps = [
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
        "//tools/test/CoverageOutputGenerator/java/com/google/devtools/coverageoutputgenerator:BranchCoverage",
        "//tools/test/CoverageOutputGenerator/java/com/google/devtools/coverageoutputgenerator:Constants",
        "//tools/test/CoverageOutputGenerator/java/com/google/devtools/coverageoutputgenerator:Coverage",
        "//tools/test/CoverageOutputGenerator/java/com/google/devtools/coverageoutputgenerator:LcovParser",
        "//tools/test/CoverageOutputGenerator/java/com/google/devtools/coverageoutputgenerator:LineCoverage",
        "//tools/test/CoverageOutputGenerator/java/com/google/devtools/coverageoutputgenerator:SourceFileCoverage",
    ],
)

java_test(
    name = "GcovParserTest",
    srcs = ["GcovParserTest.java"],
    deps = [
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
        "//tools/test/CoverageOutputGenerator/java/com/google/devtools/coverageoutputgenerator:BranchCoverage",
        "//tools/test/CoverageOutputGenerator/java/com/google/devtools/coverageoutputgenerator:Constants",
        "//tools/test/CoverageOutputGenerator/java/com/google/devtools/coverageoutputgenerator:Coverage",
        "//tools/test/CoverageOutputGenerator/java/com/google/devtools/coverageoutputgenerator:GcovParser",
        "//tools/test/CoverageOutputGenerator/java/com/google/devtools/coverageoutputgenerator:LineCoverage",
        "//tools/test/CoverageOutputGenerator/java/com/google/devtools/coverageoutputgenerator:SourceFileCoverage",
    ],
)

java_test(
    name = "CoverageTest",
    srcs = ["CoverageTest.java"],
    deps = [
        ":LcovMergerTestUtils",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
        "//tools/test/CoverageOutputGenerator/java/com/google/devtools/coverageoutputgenerator:BranchCoverage",
        "//tools/test/CoverageOutputGenerator/java/com/google/devtools/coverageoutputgenerator:Coverage",
        "//tools/test/CoverageOutputGenerator/java/com/google/devtools/coverageoutputgenerator:LineCoverage",
        "//tools/test/CoverageOutputGenerator/java/com/google/devtools/coverageoutputgenerator:SourceFileCoverage",
    ],
)

java_test(
    name = "LcovMergerFlagsTest",
    srcs = ["LcovMergerFlagsTest.java"],
    deps = [
        "//third_party:junit4",
        "//third_party:truth",
        "//tools/test/CoverageOutputGenerator/java/com/google/devtools/coverageoutputgenerator:LcovMergerFlags",
    ],
)

java_test(
    name = "MainTest",
    srcs = ["MainTest.java"],
    deps = [
        ":LcovMergerTestUtils",
        "//third_party:junit4",
        "//third_party:truth",
        "//tools/test/CoverageOutputGenerator/java/com/google/devtools/coverageoutputgenerator:Constants",
        "//tools/test/CoverageOutputGenerator/java/com/google/devtools/coverageoutputgenerator:Coverage",
        "//tools/test/CoverageOutputGenerator/java/com/google/devtools/coverageoutputgenerator:LcovParser",
        "//tools/test/CoverageOutputGenerator/java/com/google/devtools/coverageoutputgenerator:LcovPrinter",
        "//tools/test/CoverageOutputGenerator/java/com/google/devtools/coverageoutputgenerator:MainLibrary",
    ],
)

java_library(
    name = "LcovMergerTestUtils",
    testonly = 1,
    srcs = ["LcovMergerTestUtils.java"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)
