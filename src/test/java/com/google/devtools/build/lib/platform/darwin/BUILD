# Darwin specific test utilities.

package(
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

licenses(["notice"])

# Utility for sending notifications. It expects one arg which is the
# notification to send using the notify_post (3) api.
cc_binary(
    name = "notifier",
    srcs = ["notifier.c"],
    tags = ["requires-darwin"],
)
