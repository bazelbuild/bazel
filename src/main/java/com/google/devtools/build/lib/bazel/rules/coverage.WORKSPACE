# External dependencies used in coverage mode.

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# To generate an updated version of CoverageOutputGenerator:
# 1. bazel build tools/test/CoverageOutputGenerator/java/com/google/devtools/coverageoutputgenerator:coverage_output_generator_zip
# 2. Copy and rename the zip file with a new version locally.
# 3. Upload the file under https://mirror.bazel.build/bazel_coverage_output_generator/releases.
http_archive(
    name = "remote_coverage_tools",
    sha256 = "3a6951051272d51613ac4c77af6ce238a3db321bf06506fde1b8866eb18a89dd",
    urls = [
        "https://mirror.bazel.build/bazel_coverage_output_generator/releases/coverage_output_generator-v2.0.zip",
    ],
)
