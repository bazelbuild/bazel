load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_java//java:defs.bzl", "java_test")
load("@rules_rust//rust:defs.bzl", "rust_library")

package(
    default_applicable_licenses = ["//:license"],
    default_testonly = 1,
    default_visibility = ["//src/test/java/com/google/devtools/build/lib:test_lib"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src/test/java/com/google/devtools/build/lib:__pkg__"],
)

rust_library(
    name = "native_test_lib_rs",
    srcs = ["native_test_lib.rs"],
    alwayslink = True,
)

cc_binary(
    name = "native_test_lib",
    linkopts = select({
        # Rust libstd system libraries (omitted by rules_rs on Windows MSVC).
        "@rules_cc//cc/compiler:msvc-cl": [
            "ntdll.lib",
            "synchronization.lib",
        ],
        "@rules_cc//cc/compiler:clang-cl": [
            "ntdll.lib",
            "synchronization.lib",
        ],
        "@platforms//os:linux": [
            "-ldl",
            "-lpthread",
        ],
        "//conditions:default": [],
    }),
    linkshared = True,
    win_def_file = "native_test_lib.def",
    deps = [
        ":native_test_lib_rs",
    ],
)

java_test(
    name = "SkymetalBridgeTest",
    srcs = ["SkymetalBridgeTest.java"],
    data = [":native_test_lib"],
    jvm_flags = [
        "--enable-native-access=ALL-UNNAMED",
        "-Dnative.test.lib=$(rlocationpath :native_test_lib)",
    ],
    test_class = "com.google.devtools.build.lib.skymetal.SkymetalBridgeTest",
    deps = [
        "//src/main/java/com/google/devtools/build/lib/skymetal:skymetal_bridge",
        "//third_party/java/junit",
        "//third_party/java/truth",
        "@rules_java//java/runfiles",
    ],
)

java_test(
    name = "SkymetalFeatureTest",
    timeout = "short",
    srcs = ["SkymetalFeatureTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/skymetal:skymetal_feature",
        "//third_party/java/guava:collect",
        "//third_party/java/junit",
        "//third_party/java/testparameterinjector",
        "//third_party/java/truth",
    ],
)
