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
    name = "FilegroupTests_lib",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/main/java/com/google/devtools/build/lib/rules:filegroup/filegroup",
        "//src/main/java/com/google/devtools/build/lib/rules/java:java-compilation",
        "//src/main/java/com/google/devtools/build/lib/util:filetype",
        "//src/test/java/com/google/devtools/build/lib/actions/util",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
        "@maven//:com_google_testparameterinjector_test_parameter_injector",
    ],
)

java_test(
    name = "FilegroupTests",
    test_class = "com.google.devtools.build.lib.AllTests",
    runtime_deps = [
        ":FilegroupTests_lib",
        "//src/test/java/com/google/devtools/build/lib:test_runner",
    ],
)
