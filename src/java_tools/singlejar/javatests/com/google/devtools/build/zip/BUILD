# Description:
#   Tests for Zip
package(default_visibility = ["//src/java_tools/singlejar:singlejar_package_group"])

java_library(
    name = "zip_tests",
    testonly = 1,
    srcs = glob(["*.java"]),
    deps = [
        "//src/java_tools/singlejar/java/com/google/devtools/build/zip",
        "//src/test/java/com/google/devtools/build/lib:testutil",
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

filegroup(
    name = "srcs",
    srcs = glob(["*.java"]) + ["BUILD"],
)
