package(default_visibility = ["//visibility:private"])

exports_files(
    ["runfiles.sh"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__pkg__"],
)

sh_library(
    name = "runfiles_sh_lib",
    srcs = ["runfiles.sh"],
    visibility = ["//visibility:public"],
)

sh_test(
    name = "runfiles_sh_test",
    srcs = select({
        "//src:windows": ["runfiles_windows_test.sh"],
        "//src:windows_msys": ["runfiles_windows_test.sh"],
        "//src:windows_msvc": ["runfiles_windows_test.sh"],
        "//conditions:default": ["runfiles_posix_test.sh"],
    }),
    deps = [":runfiles_sh_lib"],
)

test_suite(
    name = "windows_tests",
    tags = [
        "-no_windows",
        "-slow",
    ],
    visibility = ["//visibility:private"],
)

test_suite(
    name = "all_windows_tests",
    tests = [
        ":windows_tests",
    ],
    visibility = ["//src:__pkg__"],
)
