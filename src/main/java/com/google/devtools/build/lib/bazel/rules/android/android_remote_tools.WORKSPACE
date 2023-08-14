load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_jar")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

# This must be kept in sync with the top-level WORKSPACE file.
maybe(
    http_archive,
    name = "android_tools",
    sha256 = "d7223664ca5b0de490f2a918c31f35cdf4f23a1446fe261d7470b8a22bd7bcf1",  # DO_NOT_REMOVE_THIS_ANDROID_TOOLS_UPDATE_MARKER
    url = "https://mirror.bazel.build/bazel_android_tools/android_tools_pkg-0.29.0.tar",
)

# This must be kept in sync with the top-level WORKSPACE file.
maybe(
    http_jar,
    name = "android_gmaven_r8",
    sha256 = "ab1379835c7d3e5f21f80347c3c81e2f762e0b9b02748ae5232c3afa14adf702",
    url = "https://maven.google.com/com/android/tools/r8/8.0.40/r8-8.0.40.jar",
)
