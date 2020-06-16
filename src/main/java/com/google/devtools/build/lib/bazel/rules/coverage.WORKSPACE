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
    sha256 = "0d6f73ed76908d7ec2840edd765a264b34036e4e3ec1c21a00421770523bcb27",
    urls = [
        "https://mirror.bazel.build/bazel_coverage_output_generator/releases/coverage_output_generator-v2.3.zip",
    ],
)
