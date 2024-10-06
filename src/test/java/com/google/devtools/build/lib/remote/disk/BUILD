load("@rules_java//java:defs.bzl", "java_test")

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

java_binary(
    name = "external_lock_helper",
    testonly = True,
    srcs = ["ExternalLockHelper.java"],
    jvm_flags = [
        # Prevent the JVM from polluting stdout and interfere with communication with the parent.
        "-Xlog:disable",
        "-Xlog:all=warning:stderr",
    ],
    main_class = "com.google.devtools.build.lib.remote.disk.ExternalLockHelper",
)

java_test(
    name = "disk",
    srcs = glob(["*.java"]),
    data = [":external_lock_helper"],
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        "//src/main/java/com/google/devtools/build/lib/remote:store",
        "//src/main/java/com/google/devtools/build/lib/remote/common",
        "//src/main/java/com/google/devtools/build/lib/remote/common:cache_not_found_exception",
        "//src/main/java/com/google/devtools/build/lib/remote/disk",
        "//src/main/java/com/google/devtools/build/lib/remote/util",
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs/bazel",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
        "//src/test/java/com/google/devtools/build/lib:test_runner",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestUtils",
        "//third_party:error_prone_annotations",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:mockito",
        "//third_party:truth",
        "@bazel_tools//tools/java/runfiles",
        "@protobuf//:protobuf_java",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_proto",
    ],
)
