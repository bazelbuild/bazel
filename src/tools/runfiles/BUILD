package(default_visibility = ["//visibility:private"])

exports_files(
    ["runfiles.sh"],
    visibility = ["//visibility:public"],
)

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

sh_library(
    name = "runfiles_sh_lib",
    srcs = ["runfiles.sh"],
    visibility = ["//visibility:public"],
)

sh_test(
    name = "runfiles_sh_test",
    srcs = select({
        "//src/conditions:windows": ["runfiles_windows_test.sh"],
        "//conditions:default": ["runfiles_posix_test.sh"],
    }),
    deps = [":runfiles_sh_lib"],
)

alias(
    name = "java-runfiles",
    actual = "//src/tools/runfiles/java/com/google/devtools/build/runfiles",
    visibility = ["//visibility:public"],
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
