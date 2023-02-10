load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_jar")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

# This must be kept in sync with the top-level WORKSPACE file.
maybe(
    http_archive,
    name = "android_tools",
    sha256 = "db3b02421ae974e0b33573f3e4f658d5f89cc9a0b42baae0ba2ac08e25c0720a",  # DO_NOT_REMOVE_THIS_ANDROID_TOOLS_UPDATE_MARKER
    url = "https://mirror.bazel.build/bazel_android_tools/android_tools_pkg-0.28.0.tar",
)

# This must be kept in sync with the top-level WORKSPACE file.
maybe(
    http_jar,
    name = "android_gmaven_r8",
    sha256 = "f77d9a9ebda9e32092eac4dd8e11644a7362dfa60ed6a3a9d0d32de570bbf524",
    url = "https://maven.google.com/com/android/tools/r8/4.0.48/r8-4.0.48.jar",
)
