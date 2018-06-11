package(
    default_visibility = [
        "//tools/test/LcovMerger/javatests/com/google/devtools/lcovmerger:__pkg__",
    ],
)

licenses(["notice"])  # Apache 2.0

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
        ":Coverage",
        ":LcovConstants",
        ":LineCoverage",
        ":SourceFileCoverage",
        "//third_party:guava",
    ],
)

java_library(
    name = "LcovConstants",
    srcs = ["LcovConstants.java"],
)

java_library(
    name = "LcovParser",
    srcs = ["LcovParser.java"],
    deps = [
        ":BranchCoverage",
        ":LcovConstants",
        ":LineCoverage",
        ":SourceFileCoverage",
    ],
)

java_library(
    name = "Coverage",
    srcs = ["Coverage.java"],
    deps = [":SourceFileCoverage"],
)

java_library(
    name = "MainLibrary",
    srcs = ["Main.java"],
    deps = [
        ":Coverage",
        ":LcovConstants",
        ":LcovParser",
        ":LcovPrinter",
        ":SourceFileCoverage",
        "//third_party:guava",
    ],
)

java_binary(
    name = "Main",
    srcs = ["Main.java"],
    main_class = "com.google.devtools.lcovmerger.Main",
    visibility = ["//visibility:public"],
    deps = [
        ":Coverage",
        ":LcovConstants",
        ":LcovParser",
        ":LcovPrinter",
        ":MainLibrary",
        ":SourceFileCoverage",
        "//third_party:guava",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)
