# Description:
#   singlejar C++ implementation.
package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__pkg__"],
)

filegroup(
    name = "embedded_tools",
    srcs = [
        "BUILD",
        "combiners.cc",
        "combiners.h",
        "diag.h",
        "input_jar.cc",
        "input_jar.h",
        "mapped_file.cc",
        "mapped_file.h",
        "mapped_file_posix.inc",
        "mapped_file_windows.inc",
        "options.cc",
        "options.h",
        "output_jar.cc",
        "output_jar.h",
        "port.cc",
        "port.h",
        "singlejar_main.cc",
        "token_stream.h",
        "transient_bytes.h",
        "zip_headers.h",
        "zlib_interface.h",
    ],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "singlejar",
    srcs = [
        "singlejar_main.cc",
    ],
    linkopts = select({
        "//src/conditions:freebsd": ["-lm"],
        "//conditions:default": [],
    }),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "options",
        "output_jar",
        "//third_party/zlib",
    ],
)

cc_binary(
    name = "singlejar_local",
    srcs = [
        "singlejar_local_main.cc",
    ],
    linkopts = select({
        "//src/conditions:freebsd": ["-lm"],
        "//conditions:default": [],
    }),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "combiners",
        "desugar_checking",
        "options",
        "output_jar",
        "//third_party/zlib",
    ],
)

cc_test(
    name = "combiners_test",
    size = "large",
    srcs = [
        "combiners_test.cc",
        ":zip_headers",
        ":zlib_interface",
    ],
    deps = [
        ":combiners",
        ":input_jar",
        "//third_party/zlib",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "desugar_checking_test",
    srcs = [
        "combiners_test.cc",
        ":zip_headers",
        ":zlib_interface",
    ],
    deps = [
        ":combiners",
        ":desugar_checking",
        ":input_jar",
        "//third_party/zlib",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "input_jar_empty_jar_test",
    srcs = [
        "input_jar_empty_jar_test.cc",
    ],
    data = [
        "data/empty.zip",
    ],
    deps = [
        ":input_jar",
        ":test_util",
        "//src/main/cpp/util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "input_jar_preambled_test",
    size = "large",
    srcs = [
        "input_jar_preambled_test.cc",
    ],
    data = [
        ":test1",
    ],
    deps = [
        ":input_jar",
        ":test_util",
        "//src/main/cpp/util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "input_jar_scan_jartool_test",
    size = "large",
    srcs = [
        "input_jar_scan_entries_test.h",
        "input_jar_scan_jartool_test.cc",
    ],
    data = [
        "@local_jdk//:jar",
        "@local_jdk//:jdk",
    ],
    # Timing out, see https://github.com/bazelbuild/bazel/issues/1555
    tags = ["manual"],
    deps = [
        ":input_jar",
        ":test_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "input_jar_scan_ziptool_test",
    size = "large",
    srcs = [
        "input_jar_scan_entries_test.h",
        "input_jar_scan_ziptool_test.cc",
    ],
    # Timing out, see https://github.com/bazelbuild/bazel/issues/1555
    tags = ["manual"],
    deps = [
        ":input_jar",
        ":test_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "input_jar_bad_jar_test",
    srcs = [
        "input_jar_bad_jar_test.cc",
    ],
    deps = [
        ":input_jar",
        ":test_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "options_test",
    srcs = [
        "options.h",
        "options_test.cc",
    ],
    deps = [
        ":options",
        ":token_stream",
        "//src/main/cpp/util",
        "@com_google_googletest//:gtest_main",
    ],
)

sh_test(
    name = "output_jar_bash_test",
    srcs = ["output_jar_shell_test.sh"],
    data = [
        ":singlejar",
        "//src/test/shell:bashunit",
        "@bazel_tools//tools/bash/runfiles",
    ],
    deps = ["//src/test/shell:bashunit"],
)

cc_test(
    name = "output_jar_simple_test",
    srcs = [
        "output_jar_simple_test.cc",
    ],
    data = [
        ":data1",
        ":data2",
        ":stored_jar",
        ":test1",
        ":test2",
        "@local_jdk//:jar",
        "@local_jdk//:jdk",
    ],
    deps = [
        ":input_jar",
        ":options",
        ":output_jar",
        ":port",
        ":test_util",
        "//src/main/cpp/util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "token_stream_test",
    srcs = [
        "token_stream_test.cc",
    ],
    deps = [
        ":test_util",
        ":token_stream",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "transient_bytes_test",
    size = "large",
    srcs = [
        "transient_bytes_test.cc",
        ":transient_bytes",
        ":zlib_interface",
    ],
    # Timing out, see https://github.com/bazelbuild/bazel/issues/1555
    tags = ["manual"],
    deps = [
        ":input_jar",
        ":test_util",
        "//third_party/zlib",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "zip_headers_test",
    size = "small",
    srcs = [
        "zip_headers_test.cc",
        ":zip_headers",
    ],
    deps = [
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "zlib_interface_test",
    srcs = [
        "zlib_interface_test.cc",
        ":zlib_interface",
    ],
    deps = [
        "//third_party/zlib",
        "@com_google_googletest//:gtest_main",
    ],
)

sh_test(
    name = "zip64_test",
    srcs = ["zip64_test.sh"],
    data = [
        ":singlejar",
        "//src/test/shell:bashunit",
        "@bazel_tools//tools/bash/runfiles",
        "@bazel_tools//tools/jdk:current_java_runtime",
        "@local_jdk//:jar",
    ],
    deps = ["//src/test/shell:bashunit"],
)

cc_library(
    name = "combiners",
    srcs = [
        "combiners.cc",
        ":transient_bytes",
        ":zip_headers",
    ],
    hdrs = ["combiners.h"],
    deps = [
        "//third_party/zlib",
    ],
)

cc_library(
    name = "desugar_checking",
    srcs = ["desugar_checking.cc"],
    hdrs = ["desugar_checking.h"],
    deps = [
        ":combiners",
        "//src/main/protobuf:desugar_deps_cc_proto",
    ],
)

cc_library(
    name = "diag",
    hdrs = ["diag.h"],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "port",
    srcs = ["port.cc"],
    hdrs = ["port.h"],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "mapped_file",
    srcs = ["mapped_file.cc"] + select({
        "//src:windows": ["mapped_file_windows.inc"],
        "//conditions:default": ["mapped_file_posix.inc"],
    }),
    hdrs = ["mapped_file.h"],
    visibility = ["//visibility:private"],
    deps = [
        ":diag",
        ":port",
    ],
)

cc_library(
    name = "input_jar",
    srcs = [
        "input_jar.cc",
    ],
    hdrs = [
        "input_jar.h",
        "zip_headers.h",
    ],
    deps = [
        ":diag",
        ":mapped_file",
    ],
)

cc_library(
    name = "options",
    srcs = [
        "options.cc",
        "options.h",
    ],
    hdrs = ["options.h"],
    deps = [
        ":diag",
        ":token_stream",
    ],
)

cc_library(
    name = "output_jar",
    srcs = [
        "output_jar.cc",
        "output_jar.h",
        ":zip_headers",
    ],
    hdrs = ["output_jar.h"],
    deps = [
        ":combiners",
        ":diag",
        ":input_jar",
        ":mapped_file",
        ":options",
        ":port",
        "//src/main/cpp/util",
        "//third_party/zlib",
    ],
)

cc_library(
    name = "test_util",
    srcs = ["test_util.cc"],
    hdrs = ["test_util.h"],
    testonly = 1,
    deps = [
        "//src/main/cpp/util",
        # TODO(laszlocsomor) Use @bazel_tools//tools/cpp/runfiles
        "//tools/cpp/runfiles",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "token_stream",
    hdrs = ["token_stream.h"],
    deps = [":diag"],
)

filegroup(
    name = "transient_bytes",
    srcs = [
        "diag.h",
        "transient_bytes.h",
        "zlib_interface.h",
        ":zip_headers",
    ],
)

filegroup(
    name = "zip_headers",
    srcs = ["zip_headers.h"],
)

filegroup(
    name = "zlib_interface",
    srcs = [
        "diag.h",
        "zlib_interface.h",
    ],
)

java_library(
    name = "test1",
    resources = [
        "options.cc",
        "zip_headers.h",
        "zlib_interface.h",
    ],
)

java_library(
    name = "test2",
    resources = [
        "token_stream.h",
        "transient_bytes.h",
    ],
)

java_library(
    name = "data1",
    resources = [
        "data/extra_file1",
        "data/extra_file2",
    ],
)

java_library(
    name = "data2",
    resources = [
        "data/extra_file1",
        "data/extra_file3",
    ],
)

genrule(
    name = "stored_jar",
    srcs = [
        "output_jar.cc",
        "@bazel_tools//tools/jdk:current_java_runtime",
    ],
    outs = ["stored.jar"],
    cmd = "$(location @local_jdk//:jar) -0cf \"$@\" $(location :output_jar.cc)",
    tools = ["@local_jdk//:jar"],
)
