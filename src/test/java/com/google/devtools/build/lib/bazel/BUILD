package(
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["*"]) + [
        "//src/test/java/com/google/devtools/build/lib/bazel/debug:srcs",
        "//src/test/java/com/google/devtools/build/lib/bazel/execlog:srcs",
        "//src/test/java/com/google/devtools/build/lib/bazel/repository:srcs",
        "//src/test/java/com/google/devtools/build/lib/bazel/rules:srcs",
    ],
    visibility = ["//src:__subpackages__"],
)

# List of all java_library rules containing tests that are platform agnostic.
java_library(
    name = "BazelTests_lib",
    exports = [
        "//src/test/java/com/google/devtools/build/lib/bazel/debug:WorkspaceRuleEventTest_lib",
        "//src/test/java/com/google/devtools/build/lib/bazel/execlog:StableSortTest_lib",
        "//src/test/java/com/google/devtools/build/lib/bazel/repository:RepositoryTests_lib",
        "//src/test/java/com/google/devtools/build/lib/bazel/repository/cache:RepositoryCacheTests_lib",
        "//src/test/java/com/google/devtools/build/lib/bazel/repository/downloader:DownloaderTestSuite_lib",
        "//src/test/java/com/google/devtools/build/lib/bazel/repository/starlark:StarlarkTests_lib",
        "//src/test/java/com/google/devtools/build/lib/bazel/rules:BazelRuleTests_lib",
        "//src/test/java/com/google/devtools/build/lib/bazel/rules/android:AndroidTests_lib",
        "//src/test/java/com/google/devtools/build/lib/bazel/rules/android/ndkcrosstools:NdkCrosstoolsTest_lib",
        "//src/test/java/com/google/devtools/build/lib/bazel/rules/genrule:GenruleTests_lib",
        "//src/test/java/com/google/devtools/build/lib/bazel/rules/java:JavaTests_lib",
        "//src/test/java/com/google/devtools/build/lib/bazel/rules/ninja:NinjaTests_lib",
    ],
)

test_suite(
    name = "windows_tests",
    tags = [
        "-no_windows",
        "-slow",
    ],
)

test_suite(
    name = "all_windows_tests",
    tests = [
        ":windows_tests",
        "//src/test/java/com/google/devtools/build/lib/bazel/debug:all_windows_tests",
        "//src/test/java/com/google/devtools/build/lib/bazel/execlog:all_windows_tests",
        "//src/test/java/com/google/devtools/build/lib/bazel/repository:all_windows_tests",
        "//src/test/java/com/google/devtools/build/lib/bazel/rules:all_windows_tests",
    ],
)
