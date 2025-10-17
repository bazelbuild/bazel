load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

package(default_visibility = ["//visibility:private"])

cc_binary(
    name = "postmark",
    srcs = ["postmark.cc"],
    deps = [
        "//third_party/absl/flags:flag",
        "//third_party/absl/flags:parse",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//tools:__pkg__"],
)
