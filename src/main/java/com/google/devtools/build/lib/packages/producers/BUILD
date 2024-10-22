load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "producers",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions:file_metadata",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/io:file_symlink_infinite_expansion_exception",
        "//src/main/java/com/google/devtools/build/lib/io:file_symlink_infinite_expansion_uniqueness_function",
        "//src/main/java/com/google/devtools/build/lib/io:inconsistent_filesystem_exception",
        "//src/main/java/com/google/devtools/build/lib/packages:globber",
        "//src/main/java/com/google/devtools/build/lib/skyframe:directory_listing_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:filesystem_keys",
        "//src/main/java/com/google/devtools/build/lib/skyframe:glob_descriptor",
        "//src/main/java/com/google/devtools/build/lib/skyframe:package_lookup_value",
        "//src/main/java/com/google/devtools/build/lib/util:pair",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)
