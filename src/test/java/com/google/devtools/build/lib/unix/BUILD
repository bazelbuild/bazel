load("@rules_java//java:defs.bzl", "java_library", "java_test")

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
    name = "UnixTests_lib",
    srcs = glob(["*.java"]),
    tags = [
        "foundations",
        # This test cannot run on Windows, because it uses native Posix
        # functions via JNI.
        "no_windows",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/unix:native_posix_files_service_impl",
        "//src/main/java/com/google/devtools/build/lib/unix:procmeminfo_parser",
        "//src/main/java/com/google/devtools/build/lib/unix:unix_file_system",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//src/main/java/com/google/devtools/build/lib/util:string",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/test/java/com/google/devtools/build/lib/vfs:FileSystemTest_lib",
        "//src/test/java/com/google/devtools/build/lib/vfs:SymlinkAwareFileSystemTest",
        "//src/test/java/com/google/devtools/build/lib/vfs/util",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:junit4",
        "//third_party:mockito",
        "//third_party:truth",
        "@maven//:com_google_testparameterinjector_test_parameter_injector",
    ],
)

java_test(
    name = "UnixTests",
    size = "medium",
    env = select({
        "@platforms//os:linux": {
            # Force the same locale as Bazel's client to test roundtripping of
            # arbitrary byte sequences through strings.
            "LC_ALL": "en_US.ISO-8859-1",
        },
        "//conditions:default": {},
    }),
    tags = ["no_windows"],
    test_class = "com.google.devtools.build.lib.AllTests",
    runtime_deps = [
        ":UnixTests_lib",
        "//src/test/java/com/google/devtools/build/lib:test_runner",
    ],
)
