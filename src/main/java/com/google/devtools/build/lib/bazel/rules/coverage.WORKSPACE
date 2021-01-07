# External dependencies used in coverage mode.

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# To generate an updated version of CoverageOutputGenerator:
# 1. bazel build tools/test/CoverageOutputGenerator/java/com/google/devtools/coverageoutputgenerator:coverage_output_generator_zip
# 2. Copy and rename the zip file with a new version locally.
# 3. Upload the file under https://mirror.bazel.build/bazel_coverage_output_generator/releases.
#
# This must be kept in sync with the top-level WORKSPACE file.
http_archive(
    name = "remote_coverage_tools",
    sha256 = "cd14f1cb4559e4723e63b7e7b06d09fcc3bd7ba58d03f354cdff1439bd936a7d",
    urls = [
        "https://mirror.bazel.build/bazel_coverage_output_generator/releases/coverage_output_generator-v2.5.zip",
    ],
)
