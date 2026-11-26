load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_java//java:defs.bzl", "java_binary", "java_test")
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

java_binary(
    name = "upcall_test_helper",
    testonly = True,
    srcs = ["UpcallTestHelper.java"],
    data = [":native_test_lib"],
    jvm_flags = [
        "--enable-native-access=ALL-UNNAMED",
        "-Dnative.test.lib=$(rlocationpath :native_test_lib)",
    ],
    main_class = "com.google.devtools.build.lib.skymetal.UpcallTestHelper",
    deps = [
        "//src/main/java/com/google/devtools/build/lib/skymetal:skymetal_bridge",
        "//src/main/java/com/google/devtools/build/lib/util:custom_exit_code_publisher",
        "//src/main/java/com/google/devtools/build/lib/util:custom_failure_detail_publisher",
        "@rules_java//java/runfiles",
    ],
)

java_test(
    name = "SkymetalBridgeTest",
    srcs = ["SkymetalBridgeTest.java"],
    data = [
        ":native_test_lib",
        ":upcall_test_helper",
    ],
    jvm_flags = [
        "--enable-native-access=ALL-UNNAMED",
        "-Dnative.test.lib=$(rlocationpath :native_test_lib)",
        "-Dupcall.test.helper=$(rlocationpath :upcall_test_helper)",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/skymetal:skymetal_bridge",
        "//src/main/java/com/google/devtools/build/lib/util:exit_code",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party/java/error_prone:annotations",
        "//third_party/java/junit",
        "//third_party/java/protobuf",
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
