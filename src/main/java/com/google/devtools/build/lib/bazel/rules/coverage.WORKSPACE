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
    sha256 = "96ac6bc9b9fbc67b532bcae562da1642409791e6a4b8e522f04946ee5cc3ff8e",
    urls = [
        "https://mirror.bazel.build/bazel_coverage_output_generator/releases/coverage_output_generator-v2.1.zip",
    ],
)
