# Description:
#   singlejar C++ implementation.
package(default_visibility = ["//src:__subpackages__"])

JAR_TOOL_PATH_COPT_TPL = "-DJAR_TOOL_PATH=\\\"external/local_jdk/bin/jar%s\\\""

JAR_TOOL_PATH_COPTS = select({
    "//src:windows": [JAR_TOOL_PATH_COPT_TPL % ".exe"],
    "//src:windows_msvc": [JAR_TOOL_PATH_COPT_TPL % ".exe"],
    "//src:windows_msys": [JAR_TOOL_PATH_COPT_TPL % ".exe"],
    "//conditions:default": [JAR_TOOL_PATH_COPT_TPL % ""],
})

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__pkg__"],
)

cc_binary(
    name = "singlejar",
    srcs = [
        "singlejar_main.cc",
    ],
    linkopts = select({
        "//src:freebsd": ["-lm"],
        "//conditions:default": [],
    }),
    linkstatic = 1,
    deps = [
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
        "//third_party:gtest",
        "//third_party/zlib",
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
        "//third_party:gtest",
    ],
)

cc_test(
    name = "input_jar_preambled_test",
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
        "//third_party:gtest",
    ],
)

cc_test(
    name = "input_jar_scan_jartool_test",
    size = "large",
    srcs = [
        "input_jar_scan_entries_test.h",
        "input_jar_scan_jartool_test.cc",
    ],
    copts = JAR_TOOL_PATH_COPTS,
    data = [
        "@local_jdk//:jar",
        "@local_jdk//:jdk",
    ],
    # Timing out, see https://github.com/bazelbuild/bazel/issues/1555
    tags = ["manual"],
    deps = [
        ":input_jar",
        ":test_util",
        "//third_party:gtest",
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
        "//third_party:gtest",
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
        "//third_party:gtest",
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
        ":test_util",
        ":token_stream",
        "//src/main/cpp/util",
        "//third_party:gtest",
    ],
)

sh_test(
    name = "output_jar_bash_test",
    srcs = ["output_jar_shell_test.sh"],
    args = [
        "src/test/shell",
        "$(location :singlejar)",
    ],
    data = [
        ":singlejar",
    ],
    deps = ["//src/test/shell:bashunit"],
)

cc_test(
    name = "output_jar_simple_test",
    srcs = [
        "output_jar_simple_test.cc",
    ],
    copts = JAR_TOOL_PATH_COPTS,
    data = [
        ":data1",
        ":data2",
        ":stored_jar",
        ":test1",
        ":test2",
        "@local_jdk//:jar",
        "@local_jdk//:jdk-default",
    ],
    deps = [
        ":input_jar",
        ":options",
        ":output_jar",
        ":test_util",
        "//src/main/cpp/util",
        "//third_party:gtest",
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
        "//third_party:gtest",
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
        "//third_party:gtest",
        "//third_party/zlib",
    ],
)

cc_test(
    name = "zip_headers_test",
    size = "small",
    srcs = [
        "zip_headers_test.cc",
        ":zip_headers",
    ],
    deps = ["//third_party:gtest"],
)

cc_test(
    name = "zlib_interface_test",
    srcs = [
        "zlib_interface_test.cc",
        ":zlib_interface",
    ],
    deps = [
        ":test_util",
        "//third_party:gtest",
        "//third_party/zlib",
    ],
)

sh_test(
    name = "zip64_test",
    srcs = ["zip64_test.sh"],
    args = [
        "src/test/shell",
        "$(location :singlejar)",
        "$(location @local_jdk//:jar)",
    ],
    data = [
        ":singlejar",
        "//tools/defaults:jdk",
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
    deps = ["//third_party/zlib"],
)

cc_library(
    name = "input_jar",
    srcs = [
        "diag.h",
        "input_jar.cc",
        "mapped_file.h",
    ],
    hdrs = [
        "input_jar.h",
        "zip_headers.h",
    ],
)

cc_library(
    name = "options",
    srcs = [
        "diag.h",
        "options.cc",
        "options.h",
    ],
    hdrs = ["options.h"],
    deps = [
        ":token_stream",
    ],
)

cc_library(
    name = "output_jar",
    srcs = [
        "diag.h",
        "mapped_file.h",
        "output_jar.cc",
        "output_jar.h",
        ":zip_headers",
    ],
    hdrs = ["output_jar.h"],
    deps = [
        ":combiners",
        ":input_jar",
        ":options",
        "//src/main/cpp/util",
        "//third_party/zlib",
    ],
)

cc_library(
    name = "test_util",
    srcs = ["test_util.cc"],
    hdrs = ["test_util.h"],
    deps = [
        "//src/main/cpp/util",
        "//third_party:gtest",
    ],
)

cc_library(
    name = "token_stream",
    srcs = ["diag.h"],
    hdrs = ["token_stream.h"],
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
        "//tools/defaults:jdk",
    ],
    outs = ["stored.jar"],
    cmd = "$(location @local_jdk//:jar) -0cf \"$@\" $(location :output_jar.cc)",
    tools = ["@local_jdk//:jar"],
)
