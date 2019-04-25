package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # MIT

filegroup(
    name = "jni_src",
    srcs = ["decoder_jni.cc"],
)

java_library(
    name = "dec",
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
        ":dec",
        "//org/brotli/integration:brotli_jni_test_base",
        "//org/brotli/integration:bundle_helper",
        "@junit_junit//jar",
    ],
)

filegroup(
    name = "brotli_jni",
    srcs = ["//:brotli_jni.dll"],
)

filegroup(
    name = "test_bundle",
    srcs = ["//org/brotli/integration:test_data"],
)

java_test(
    name = "BrotliDecoderChannelTest",
    test_class = "org.brotli.wrapper.dec.BrotliDecoderChannelTest",
    size = "large",
    data = [
        ":brotli_jni",  # Bazel JNI workaround
        ":test_bundle",
    ],
    jvm_flags = [
        "-DBROTLI_JNI_LIBRARY=$(location :brotli_jni)",
        "-DTEST_BUNDLE=$(location :test_bundle)",
    ],
    runtime_deps = [":test_lib"],
)

java_test(
    name = "BrotliInputStreamTest",
    test_class = "org.brotli.wrapper.dec.BrotliInputStreamTest",
    size = "large",
    data = [
        ":brotli_jni",  # Bazel JNI workaround
        ":test_bundle",
    ],
    jvm_flags = [
        "-DBROTLI_JNI_LIBRARY=$(location :brotli_jni)",
        "-DTEST_BUNDLE=$(location :test_bundle)",
    ],
    runtime_deps = [":test_lib"],
)

java_test(
    name = "DecoderTest",
    test_class = "org.brotli.wrapper.dec.DecoderTest",
    size = "large",
    data = [
        ":brotli_jni",  # Bazel JNI workaround
        ":test_bundle",
    ],
    jvm_flags = [
        "-DBROTLI_JNI_LIBRARY=$(location :brotli_jni)",
        "-DTEST_BUNDLE=$(location :test_bundle)",
    ],
    runtime_deps = [":test_lib"],
)
