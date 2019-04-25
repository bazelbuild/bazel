package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # MIT

filegroup(
    name = "jni_src",
    srcs = ["common_jni.cc"],
)

filegroup(
    name = "brotli_jni_no_dictionary_data",
    srcs = ["//:brotli_jni_no_dictionary_data.dll"],
)

java_library(
    name = "common",
    srcs = glob(
        ["*.java"],
        exclude = ["*Test*.java"],
    ),
)

java_library(
    name = "test_lib",
    testonly = 1,
    srcs = glob(["*Test*.java"]),
    deps = [
        ":common",
        "//org/brotli/dec",
        "//org/brotli/integration:brotli_jni_test_base",
        "//org/brotli/wrapper/dec",
        "@junit_junit//jar",
    ],
)

java_test(
    name = "SetZeroDictionaryTest",
    test_class = "org.brotli.wrapper.common.SetZeroDictionaryTest",
    size = "small",
    data = [
        ":brotli_jni_no_dictionary_data",  # Bazel JNI workaround
    ],
    jvm_flags = [
        "-DBROTLI_JNI_LIBRARY=$(location :brotli_jni_no_dictionary_data)",
    ],
    runtime_deps = [":test_lib"],
)

java_test(
    name = "SetRfcDictionaryTest",
    test_class = "org.brotli.wrapper.common.SetRfcDictionaryTest",
    size = "small",
    data = [
        ":brotli_jni_no_dictionary_data",  # Bazel JNI workaround
    ],
    jvm_flags = [
        "-DBROTLI_JNI_LIBRARY=$(location :brotli_jni_no_dictionary_data)",
    ],
    runtime_deps = [":test_lib"],
)
