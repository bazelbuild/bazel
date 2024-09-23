load("@rules_java//java:defs.bzl", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_testonly = 1,
    default_visibility = ["//src/tools/remote:__pkg__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_test(
    name = "http",
    srcs = glob(["*.java"]),
    test_class = "com.google.devtools.build.remote.worker.http.AbstractHttpCacheServerHandlerTest",
    deps = [
        "//src/tools/remote/src/main/java/com/google/devtools/build/remote/worker/http",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)
