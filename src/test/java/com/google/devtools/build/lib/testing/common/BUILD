load("@rules_java//java:defs.bzl", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_testonly = True,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = False,
    srcs = glob(["**"]),
    visibility = ["//src/test/java/com/google/devtools/build/lib:__pkg__"],
)

java_test(
    name = "DirectoryListingHelperTest",
    srcs = ["DirectoryListingHelperTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/testing/common:directory_listing_helper",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//third_party/java/junit",
        "//third_party/java/truth",
    ],
)

java_test(
    name = "FakeOptionsTest",
    srcs = ["FakeOptionsTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/testing/common:fake-options",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party/java/junit",
        "//third_party/java/truth",
    ],
)
