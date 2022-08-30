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
    sha256 = "23e23daae0ab4aba5eb30b0100bb3192aef7274f44a38f45a69ad52a08f15b16",
    url = "https://maven.google.com/com/android/tools/r8/3.0.64/r8-3.0.64.jar",
)
