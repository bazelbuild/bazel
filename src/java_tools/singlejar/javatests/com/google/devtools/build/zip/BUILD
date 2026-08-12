load("@rules_java//java:defs.bzl", "java_library", "java_test")

# Description:
#   Tests for Zip
package(
    default_applicable_licenses = ["//:license"],
)

java_library(
    name = "zip_tests",
    testonly = 1,
    srcs = glob(["*.java"]),
    deps = [
        "//src/java_tools/singlejar/java/com/google/devtools/build/zip",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestSuite",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "ZipTests",
    test_class = "com.google.devtools.build.zip.ZipTests",
    runtime_deps = [":zip_tests"],
)

# DOS timestamps are always interpreted in the local time zone, so packaging is exercised on both
# sides of UTC: behind it, an instant pinned to UTC midnight falls before the DOS epoch and is not
# representable at all, ahead of it, it records a later time of day than on a UTC machine.
java_test(
    name = "PackageZipperTest_behind_utc",
    env = {"TZ": "America/Los_Angeles"},
    test_class = "com.google.devtools.build.zip.PackageZipperTest",
    runtime_deps = [":zip_tests"],
)

java_test(
    name = "PackageZipperTest_ahead_of_utc",
    env = {"TZ": "Europe/Berlin"},
    test_class = "com.google.devtools.build.zip.PackageZipperTest",
    runtime_deps = [":zip_tests"],
)

filegroup(
    name = "srcs",
    srcs = glob(["*.java"]) + ["BUILD"],
    visibility = ["//src:__subpackages__"],
)
