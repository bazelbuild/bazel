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
    visibility = ["//src:__subpackages__"],
)

java_test(
    name = "DirectoryListingHelperTest",
    srcs = ["DirectoryListingHelperTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/testing/common:directory_listing_helper",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "FakeOptionsTest",
    srcs = ["FakeOptionsTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/testing/common:fake-options",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)
