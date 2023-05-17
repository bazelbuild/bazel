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
    name = "credential_module",
    srcs = [
        "CredentialModule.java",
        "SystemMillisTicker.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib/authandtls",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:caffeine",
        "//third_party:guava",
    ],
)

java_library(
    name = "credentialhelper",
    srcs = glob(
        ["*.java"],
        exclude = [
            "CredentialModule.java",
            "SystemMillisTicker.java",
        ],
    ),
    deps = [
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party:auth",
        "//third_party:auto_value",
        "//third_party:caffeine",
        "//third_party:error_prone_annotations",
        "//third_party:gson",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)
