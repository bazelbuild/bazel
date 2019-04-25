package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # MIT

filegroup(
    name = "jni_src",
    srcs = ["encoder_jni.cc"],
)

filegroup(
    name = "brotli_jni",
    srcs = ["//:brotli_jni.dll"],
)

java_library(
    name = "enc",
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
        ":enc",
        "//org/brotli/integration:brotli_jni_test_base",
        "//org/brotli/integration:bundle_helper",
        "//org/brotli/wrapper/dec",
        "@junit_junit//jar",
    ],
)

filegroup(
    name = "test_bundle",
    srcs = ["//org/brotli/integration:test_corpus"],
)

java_test(
    name = "BrotliEncoderChannelTest",
    test_class = "org.brotli.wrapper.enc.BrotliEncoderChannelTest",
    size = "large",
    data = [
        ":brotli_jni",  # Bazel JNI workaround
        ":test_bundle",
    ],
    jvm_flags = [
        "-DBROTLI_JNI_LIBRARY=$(location :brotli_jni)",
        "-DTEST_BUNDLE=$(location :test_bundle)",
    ],
    shard_count = 15,
    runtime_deps = [":test_lib"],
)

java_test(
    name = "BrotliOutputStreamTest",
    test_class = "org.brotli.wrapper.enc.BrotliOutputStreamTest",
    size = "large",
    data = [
        ":brotli_jni",  # Bazel JNI workaround
        ":test_bundle",
    ],
    jvm_flags = [
        "-DBROTLI_JNI_LIBRARY=$(location :brotli_jni)",
        "-DTEST_BUNDLE=$(location :test_bundle)",
    ],
    shard_count = 15,
    runtime_deps = [":test_lib"],
)

java_test(
    name = "EncoderTest",
    test_class = "org.brotli.wrapper.enc.EncoderTest",
    size = "large",
    data = [
        ":brotli_jni",  # Bazel JNI workaround
        ":test_bundle",
    ],
    jvm_flags = [
        "-DBROTLI_JNI_LIBRARY=$(location :brotli_jni)",
        "-DTEST_BUNDLE=$(location :test_bundle)",
    ],
    shard_count = 15,
    runtime_deps = [":test_lib"],
)
