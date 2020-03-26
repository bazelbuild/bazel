load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_jar")

# This must be kept in sync with the top-level WORKSPACE file.
http_archive(
    name = "android_tools",
    sha256 = "e2cbd43a9d23aa32197c29d689a7e017f205acb07053f5dd584f500a1a9d4361",
    url = "https://mirror.bazel.build/bazel_android_tools/android_tools_pkg-0.16.0.tar.gz",
)

# This must be kept in sync with the top-level WORKSPACE file.
http_jar(
    name = "android_gmaven_r8",
    sha256 = "824d94de0210df3692a812e18ffa334a17365e059856ae03c772e9490d61d4d8",
    url = "https://maven.google.com/com/android/tools/r8/1.6.84/r8-1.6.84.jar",
)
