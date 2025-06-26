load("@rules_shell//shell:sh_library.bzl", "sh_library")
load("@rules_shell//shell:sh_test.bzl", "sh_test")

package(default_visibility = ["//visibility:private"])

filegroup(
    name = "srcs",
    srcs = glob(
        ["**"],
        exclude = [
            "*~",
            ".*",
        ],
    ),
    visibility = ["//tools/bash:__pkg__"],
)

filegroup(
    name = "embedded_tools",
    srcs = [
        "BUILD.tools",
        "runfiles.bash",
    ],
    visibility = ["//tools/bash:__pkg__"],
)

sh_library(
    name = "runfiles_lib",
    testonly = 1,
    srcs = ["runfiles.bash"],
)

sh_test(
    name = "runfiles_test",
    srcs = ["runfiles_test.bash"],
    deps = [":runfiles_lib"],
)
