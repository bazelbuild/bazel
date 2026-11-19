load("@rules_java//java:defs.bzl", "java_library", "java_test")
load("//src:java_opt_binary.bzl", "java_opt_binary")

package(
    default_applicable_licenses = ["//:license"],
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "BazelTests_lib",
    srcs = glob(
        [
            "*.java",
        ],
        exclude = [
            "BazelHashFunctionsBenchmark.java",
        ],
    ),
    deps = [
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs/bazel",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//third_party:gson",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:junit4",
        "//third_party:truth",
        "//third_party/protobuf:protobuf_java",
        "@rules_java//java/runfiles",
    ],
)

java_test(
    name = "BazelTests",
    size = "small",
    data = ["@blake3_test_vectors//file"],
    jvm_flags = ["-Dblake3.test_vectors=$(rlocationpath @blake3_test_vectors//file)"],
    test_class = "com.google.devtools.build.lib.AllTests",
    runtime_deps = [
        ":BazelTests_lib",
        "//src/test/java/com/google/devtools/build/lib:AllTests",
    ],
)

java_opt_binary(
    name = "BazelHashFunctionsBenchmark",
    srcs = ["BazelHashFunctionsBenchmark.java"],
    jvm_flags = ["--enable-native-access=ALL-UNNAMED"],
    main_class = "org.openjdk.jmh.Main",
    deps = [
        "//src/main/java/com/google/devtools/build/lib/vfs/bazel",
        "//third_party:guava",
        "//third_party:jmh",
    ],
)
