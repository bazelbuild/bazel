# This package contains sources for language-specific runfiles libraries.
#
# Do not depend on these rules. Depend on the ones in @bazel_tools//tools/runfiles instead, that way
# your binary won't require the Bazel source tree to work correctly.

package(default_visibility = ["//visibility:private"])

filegroup(
    name = "srcs",
    srcs = glob(
        ["**"],
        exclude = [".*"],  # .swp files and such
    ) + [
        "//src/tools/runfiles/java/com/google/devtools/build/runfiles:srcs",
    ],
    visibility = ["//src:__pkg__"],
)

filegroup(
    name = "embedded_tools",
    srcs = [
        "BUILD.tools",
        "//src/tools/runfiles/java/com/google/devtools/build/runfiles:embedded_tools",
    ],
    visibility = ["//src:__pkg__"],
)

test_suite(
    name = "windows_tests",
    tags = [
        "-no_windows",
        "-slow",
    ],
)

test_suite(
    name = "all_windows_tests",
    tests = [
        ":windows_tests",
        "//src/tools/runfiles/java/com/google/devtools/build/runfiles:all_windows_tests",
    ],
    visibility = ["//src:__pkg__"],
)
