load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_jar")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

# This must be kept in sync with the top-level WORKSPACE file.
maybe(
    http_archive,
    name = "android_tools",
    sha256 = "5d0f140125afba82603ccd5050c78dd2e2863ca992a17f43f6df9a9119ffcb9b",  # DO_NOT_REMOVE_THIS_ANDROID_TOOLS_UPDATE_MARKER
    url = "https://mirror.bazel.build/bazel_android_tools/android_tools_pkg-0.27.2.tar",
)

# This must be kept in sync with the top-level WORKSPACE file.
maybe(
    http_jar,
    name = "android_gmaven_r8",
    sha256 = "ab1379835c7d3e5f21f80347c3c81e2f762e0b9b02748ae5232c3afa14adf702",
    url = "https://maven.google.com/com/android/tools/r8/8.0.40/r8-8.0.40.jar",
)
