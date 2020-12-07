load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load("@bazel_tools//tools/cpp:cc_configure.bzl", "cc_configure")
load("//:distdir_deps.bzl", "DIST_DEPS")

# rules_cc is used in @bazel_tools//tools/cpp, so must be loaded here.
maybe(
    http_archive,
    "rules_cc",
    sha256 = DIST_DEPS["rules_cc"]["sha256"],
    strip_prefix = DIST_DEPS["rules_cc"]["strip_prefix"],
    urls = DIST_DEPS["rules_cc"]["urls"],
)

cc_configure()
