load("@rules_java//java:defs.bzl", "java_library")

package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "bugreport",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis:blaze_version_info",
        "//src/main/java/com/google/devtools/build/lib/util:TestType",
        "//src/main/java/com/google/devtools/build/lib/util:crash_failure_details",
        "//src/main/java/com/google/devtools/build/lib/util:custom_exit_code_publisher",
        "//src/main/java/com/google/devtools/build/lib/util:custom_failure_detail_publisher",
        "//src/main/java/com/google/devtools/build/lib/util:detailed_exit_code",
        "//src/main/java/com/google/devtools/build/lib/util:exit_code",
        "//src/main/java/com/google/devtools/build/lib/util:logging",
        "//src/main/java/com/google/devtools/build/lib/util/io:out-err",
        "//third_party:flogger",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)
