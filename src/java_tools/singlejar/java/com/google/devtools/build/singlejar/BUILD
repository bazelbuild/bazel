# Description:
#   SingleJar combines multiple zip files and additional files
#   into a single zip file.
package(default_visibility = ["//src/java_tools/singlejar:singlejar_package_group"])

filegroup(
    name = "srcs",
    srcs = glob(["*"]),
)

# This is used as a library outside of Bazel.
java_library(
    name = "libSingleJar",
    srcs = [
        "ConcatenateStrategy.java",
        "CopyEntryFilter.java",
        "DefaultJarEntryFilter.java",
        "JarUtils.java",
        "ZipCombiner.java",
        "ZipEntryFilter.java",
    ],
    visibility = [
        "//src/java_tools/singlejar:__pkg__",
        "//src/java_tools/singlejar:singlejar_package_group",
        "//src/java_tools/singlejar/javatests/com/google/devtools/build/singlejar:__pkg__",
        "//src/test/java/com/google/devtools/build/android:__pkg__",
        "//src/tools/android/java/com/google/devtools/build/android:__pkg__",
    ],
    deps = [
        "//src/java_tools/singlejar/java/com/google/devtools/build/zip",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "libSingleJarMain",
    srcs = glob(["**/*.java"]),
    # Avoid adding dependencies here - this is a very low-level library and we don't want to pull in
    # the world, even including commons.
    deps = [
        "//src/java_tools/singlejar/java/com/google/devtools/build/zip",
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//src/main/protobuf:worker_protocol_java_proto",
        "//third_party:jsr305",
    ],
)

java_binary(
    name = "bazel-singlejar",
    srcs = glob(["*.java"]),
    main_class = "com.google.devtools.build.singlejar.SingleJar",
    visibility = [
        "//:__subpackages__",
        "//src/java_tools/singlejar:singlejar_package_group",
    ],
    deps = [
        "//src/java_tools/singlejar/java/com/google/devtools/build/zip",
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//src/main/protobuf:worker_protocol_java_proto",
        "//third_party:jsr305",
    ],
)

#
# Bootstrapping using Skylark rules
#

load("//tools/build_rules:java_rules_skylark.bzl", "bootstrap_java_library", "bootstrap_java_binary")

bootstrap_java_library(
    name = "skylark-deps",
    jars = [
        "//third_party:jsr305-jars",
    ],
    tags = ["manual"],
)

bootstrap_java_binary(
    name = "bootstrap",
    srcs = glob(
        ["**/*.java"],
        exclude = ["**/SingleJarWorker.java"],
    ) + [
        "//src/java_tools/singlejar/java/com/google/devtools/build/zip:java-srcs",
    ],
    main_class = "com.google.devtools.build.singlejar.SingleJar",
    tags = ["manual"],
    visibility = ["//visibility:public"],
    deps = [
        ":skylark-deps",
        "//src/main/java/com/google/devtools/build/lib/shell:shell-skylark",
    ],
)
