package(default_visibility = ["//visibility:private"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__pkg__"],
)

filegroup(
    name = "test-deps",
    testonly = 1,
    srcs = ["//src:bazel_with_jdk"],
)

py_library(
    name = "test_base",
    testonly = 1,
    srcs = ["test_base.py"],
    data = [":test-deps"],
    visibility = [
        "//src/test/py/bazel:__pkg__",
        "//third_party/def_parser:__pkg__",
        "//tools/android:__pkg__",
    ],
)

py_test(
    name = "action_temp_test",
    size = "medium",
    srcs = ["action_temp_test.py"],
    deps = [":test_base"],
)

py_test(
    name = "cc_import_test",
    size = "medium",
    srcs = ["cc_import_test.py"],
    deps = [":test_base"],
)

py_test(
    name = "bazel_server_mode_test",
    size = "medium",
    srcs = ["bazel_server_mode_test.py"],
    deps = [":test_base"],
)

py_test(
    name = "bazel_clean_test",
    size = "medium",
    srcs = ["bazel_clean_test.py"],
    deps = [":test_base"],
)

py_test(
    name = "bazel_external_repository_test",
    size = "medium",
    srcs = ["bazel_external_repository_test.py"],
    tags = ["requires_internet"],
    deps = [":test_base"],
)

py_test(
    name = "bazel_windows_test",
    size = "medium",
    srcs = select({
        "//src/conditions:windows": ["bazel_windows_test.py"],
        "//src/conditions:windows_msvc": ["bazel_windows_test.py"],
        "//src/conditions:windows_msys": ["bazel_windows_test.py"],
        "//conditions:default": ["empty_test.py"],
    }),
    main = select({
        "//src/conditions:windows": "bazel_windows_test.py",
        "//src/conditions:windows_msvc": "bazel_windows_test.py",
        "//src/conditions:windows_msys": "bazel_windows_test.py",
        "//conditions:default": "empty_test.py",
    }),
    deps = select({
        "//src/conditions:windows": [":test_base"],
        "//src/conditions:windows_msvc": [":test_base"],
        "//src/conditions:windows_msys": [":test_base"],
        "//conditions:default": [],
    }),
)

py_test(
    name = "launcher_test",
    size = "medium",
    srcs = ["launcher_test.py"],
    deps = [":test_base"],
)

py_test(
    name = "runfiles_test",
    srcs = ["runfiles_test.py"],
    deps = [":test_base"],
)

py_test(
    name = "bazel_windows_cpp_test",
    size = "large",
    srcs = select({
        "//src/conditions:windows": ["bazel_windows_cpp_test.py"],
        "//src/conditions:windows_msvc": ["bazel_windows_cpp_test.py"],
        "//src/conditions:windows_msys": ["bazel_windows_cpp_test.py"],
        "//conditions:default": ["empty_test.py"],
    }),
    main = select({
        "//src/conditions:windows": "bazel_windows_cpp_test.py",
        "//src/conditions:windows_msvc": "bazel_windows_cpp_test.py",
        "//src/conditions:windows_msys": "bazel_windows_cpp_test.py",
        "//conditions:default": "empty_test.py",
    }),
    deps = select({
        "//src/conditions:windows": [":test_base"],
        "//src/conditions:windows_msvc": [":test_base"],
        "//src/conditions:windows_msys": [":test_base"],
        "//conditions:default": [],
    }),
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
