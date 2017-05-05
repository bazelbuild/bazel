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
        "//third_party/java/jacoco:blaze-agent",
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

NOP_JARJAR_COMMAND = "cp \"$(location :JacocoCoverage_deploy.jar)\" \"$@\""

genrule(
    name = "Jacoco_jarjar",
    srcs = [
        ":JacocoCoverage_deploy.jar",
        ":JacocoCoverage.jarjar",
    ],
    outs = ["JacocoCoverage_jarjar_deploy.jar"],
    cmd = select({
        # Bazel 0.4.3 contains two bugs: a quoting bug in the Java cmd.exe
        # wrapper script that makes it unable to handle $ signs in paths (#2306)
        # and one that makes it occasionally put $ signs in the output base
        # (#2342).
        #
        # These two collude to make it impossible to run built Java binaries on
        # Windows if the output base happens to contain a $ sign.
        #
        # Thus, don't call jarjar when on Windows. This makes Java coverage not
        # work if the code under test uses libraries the test runner also does
        # (e.g. ASM).
        #
        # TODO(lberki): Remove this once a Bazel version with a fix to either of
        # these bugs is out.
        #
        # Additionally, there is a hard-coded path to /bin/bash which is wrong on
        # freebsd; so disable there as well until a fixed version is in there as
        # well.
        # TODO(aehlig): fix and remove once a version with the fix is out.
        #
        # We don't invoke jarjar_bin directly because this command is invoked
        # during bootstrapping when we don't have build-runfiles, thus no
        # runfiles trees. The Java launcher script looks in the runfiles tree
        # for the jars (and rightfully so), thus, invoking the binary directly
        # won't work.
        "//src:freebsd": NOP_JARJAR_COMMAND,
        "//src:windows": NOP_JARJAR_COMMAND,
        "//src:windows_msvc": NOP_JARJAR_COMMAND,
        "//src:windows_msys": NOP_JARJAR_COMMAND,
        "//conditions:default": "\n".join([
            "JARJAR=\"$$(mktemp)\"",
            "\"$(JAVA)\" -jar \"$(location //third_party/java/jarjar:jarjar_bin_deploy.jar)\" --rules \"$(location :JacocoCoverage.jarjar)\" --output \"$${JARJAR}\" \"$(location :JacocoCoverage_deploy.jar)\"",
            "$(location //src/tools/singlejar:singlejar) --normalize --sources \"$${JARJAR}\" --output \"$@\"",
            "rm -fr \"$${JARJAR}\"",
        ]),
    }),
    tools = select({
        "//src:freebsd": [],
        "//src:windows": [],
        "//conditions:default": [
            "//third_party/java/jarjar:jarjar_bin_deploy.jar",
            "//tools/defaults:jdk",
            "//src/tools/singlejar",
        ],
    }),
)
