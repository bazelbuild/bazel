load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

ANDROID_TOOLS_COMMIT = "1a6aaad88aaaf186c168623c872c55ed80c05c5a"

http_archive(
    name = "legacy_android_tools",
    url = "https://github.com/jin/legacy_android_tools/archive/%s.zip" % ANDROID_TOOLS_COMMIT,
    strip_prefix = "legacy_android_tools-%s" % ANDROID_TOOLS_COMMIT,
    sha256 = "",
)
