package(default_visibility = ["//visibility:private"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__pkg__"],
)

filegroup(
    name = "test-deps",
    testonly = 1,
    srcs = ["//src:bazel"],
)

py_library(
    name = "test_base",
    testonly = 1,
    srcs = ["test_base.py"],
    data = [":test-deps"],
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
        "//src:windows": ["bazel_windows_test.py"],
        "//src:windows_msvc": ["bazel_windows_test.py"],
        "//src:windows_msys": ["bazel_windows_test.py"],
        "//conditions:default": ["empty_test.py"],
    }),
    main = select({
        "//src:windows": "bazel_windows_test.py",
        "//src:windows_msvc": "bazel_windows_test.py",
        "//src:windows_msys": "bazel_windows_test.py",
        "//conditions:default": "empty_test.py",
    }),
    deps = select({
        "//src:windows": [":test_base"],
        "//src:windows_msvc": [":test_base"],
        "//src:windows_msys": [":test_base"],
        "//conditions:default": [],
    }),
)

py_test(
    name = "launcher_test",
    size = "medium",
    srcs = ["launcher_test.py"],
    deps = [":test_base"],
)
