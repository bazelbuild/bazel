package(
    default_visibility = ["//visibility:legacy_public"],
)

licenses(["notice"])

filegroup(
    name = "bazel-srcs",
    testonly = 0,
    srcs = glob(
        ["**"],
        exclude = ["BUILD"],
    ),
    visibility = ["//third_party/bazel:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

filegroup(
    name = "embedded_tools",
    srcs = [
        "JacocoCoverage_jarjar_deploy.jar",
    ],
)

filegroup(
    name = "jacoco_coverage_runtime",
    srcs = ["JacocoCoverage_deploy.jar"],
)

# Bazel custom Jacoco runner used to provide proper initialization and lcov
# report generation when using offline Jacoco instrumentation.
# This target should not be used as a dependency (except when writing tests for
# it).
#
# An implicit dependency of all "java_binary" rules.
java_binary(
    name = "JacocoCoverage",
    srcs = [
        "BranchCoverageDetail.java",
        "BranchDetailAnalyzer.java",
        "BranchExp.java",
        "ClassProbesMapper.java",
        "CovExp.java",
        "JacocoCoverageRunner.java",
        "JacocoLCOVFormatter.java",
        "MethodProbesMapper.java",
        "ProbeExp.java",
    ],
    deps = [
        ":bitfield",
        "//third_party/java/jacoco:blaze-agent-neverlink",
        "//third_party/java/jacoco:core",
        "//third_party/java/jacoco:report",
    ],
)

java_library(
    name = "bitfield",
    srcs = [
        "BitField.java",
        "IllegalStringException.java",
    ],
    deps = [
        "//third_party:apache_commons_lang",
    ],
)

genrule(
    name = "Jacoco_jarjar",
    srcs = [
        ":JacocoCoverage_deploy.jar",
        ":JacocoCoverage.jarjar",
    ],
    outs = ["JacocoCoverage_jarjar_deploy.jar"],
    cmd = "\n".join([
        # File#createTempFile() apparently tries to create the temp file under
        # C:\WINDOWS by default on Windows, which is of course wrong. Help Java a bit.
        # Remove this once https://github.com/bazelbuild/bazel/issues/2349 is fixed.
        "if [[ $$(uname -a) =~ MSYS ]]; then",
        "  export TEMP=\"$$(cygpath -m --mydocs)/temp\";",
        "  mkdir -p \"$$(cygpath -u --mydocs)/temp\";",
        "fi",
        "$(location //third_party/java/jarjar:jarjar_bin) process $(location :JacocoCoverage.jarjar) $(location :JacocoCoverage_deploy.jar) $@",
    ]),
    tools = ["//third_party/java/jarjar:jarjar_bin"],
)
