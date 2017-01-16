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

genrule(
    name = "Jacoco_jarjar",
    srcs = [
        ":JacocoCoverage_deploy.jar",
        ":JacocoCoverage.jarjar",
    ],
    outs = ["JacocoCoverage_jarjar_deploy.jar"],
    cmd = "\n".join([
        # File#createTempFile() apparently tries to create the temp file under
        # C:\WINDOWS by default on Windows, which is not writable. Help Java a bit.
        # https://github.com/bazelbuild/bazel/issues/2349 will be a general
        # solution for creating temp files in Java binaries in actions. Once
        # that's fixed, we won't need to set TEMP here anymore.
        "set +e",
        "if [[ $$(uname -a) =~ MSYS ]]; then",
        "  DOS_TEMP=\"$$(cygpath -m --mydocs)/temp\";",
        "  PATH_DOS=$$?;",
        "  UNIX_TEMP=\"$$(cygpath -u --mydocs)/temp\";",
        "  if [[ $$? != 0 || $$PATH_DOS != 0 ]]; then",
        "    echo Cannot find temp directory, defaulting to /c/temp;",
        "    export DOS_TEMP=C:\\\\TEMP;",
        "    UNIX_TEMP=/c/temp;",
        "  fi",
        "  export TEMP=\"$$DOS_TEMP\";",
        "  mkdir -p \"$$UNIX_TEMP\";",
        "fi",
        "\"$(location //third_party/java/jarjar:jarjar_bin)\" process \"$(location :JacocoCoverage.jarjar)\" \"$(location :JacocoCoverage_deploy.jar)\" \"$@\"",
    ]),
    tools = ["//third_party/java/jarjar:jarjar_bin"],
)
