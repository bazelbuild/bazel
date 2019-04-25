# Description:
#   Java port of Brotli decoder.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # MIT

java_library(
    name = "dec",
    srcs = glob(
        ["*.java"],
        exclude = ["*Test*.java"],
    ),
    proguard_specs = ["proguard.cfg"],
)

java_library(
    name = "test_lib",
    testonly = 1,
    srcs = glob(["*Test*.java"]),
    deps = [
        ":dec",
        "@junit_junit//jar",
    ],
)

java_test(
    name = "BitReaderTest",
    test_class = "org.brotli.dec.BitReaderTest",
    runtime_deps = [":test_lib"],
)

java_test(
    name = "DecodeTest",
    test_class = "org.brotli.dec.DecodeTest",
    runtime_deps = [":test_lib"],
)

java_test(
    name = "DictionaryTest",
    test_class = "org.brotli.dec.DictionaryTest",
    runtime_deps = [":test_lib"],
)

java_test(
    name = "EagerStreamTest",
    test_class = "org.brotli.dec.EagerStreamTest",
    runtime_deps = [":test_lib"],
)

java_test(
    name = "SynthTest",
    test_class = "org.brotli.dec.SynthTest",
    runtime_deps = [":test_lib"],
)

java_test(
    name = "TransformTest",
    test_class = "org.brotli.dec.TransformTest",
    runtime_deps = [":test_lib"],
)
