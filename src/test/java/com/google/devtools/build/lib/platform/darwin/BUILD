# Darwin specific test utilities.

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

package(
    default_applicable_licenses = ["//:license"],
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

# Utility for sending notifications. It expects one arg which is the
# notification to send using the notify_post (3) api.
cc_binary(
    name = "notifier",
    srcs = ["notifier.c"],
    tags = ["requires-darwin"],
    target_compatible_with = ["@platforms//os:macos"],
)
