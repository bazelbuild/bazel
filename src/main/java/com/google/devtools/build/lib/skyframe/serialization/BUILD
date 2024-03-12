load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec:srcs",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/testutils:srcs",
    ],
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "codec-scanning-constants",
    srcs = ["CodecScanningConstants.java"],
)

java_library(
    name = "serialization",
    srcs = glob(
        ["**/*.java"],
        exclude = [
            "SerializationConstants.java",
            "CodecScanningConstants.java",
            "VisibleForSerialization.java",
        ],
    ),
    add_opens = [
        "java.base/java.lang.invoke",
    ],
    deps = [
        ":codec-scanning-constants",
        ":constants",
        "//src/main/java/com/google/devtools/build/lib/bugreport",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec:registered-singleton",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec:serialization-constant",
        "//src/main/java/com/google/devtools/build/lib/unsafe:string",
        "//src/main/java/com/google/devtools/build/lib/unsafe:unsafe-provider",
        "//third_party:auto_value",
        "//third_party:caffeine",
        "//third_party:error_prone_annotations",
        "//third_party:fastutil",
        "//third_party:flogger",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/protobuf:protobuf_java",
    ],
)

java_library(
    name = "constants",
    srcs = ["SerializationConstants.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/util:TestType",
        "//src/main/java/com/google/devtools/build/lib/util:resource_usage",
    ],
)

java_library(
    name = "visible-for-serialization",
    srcs = ["VisibleForSerialization.java"],
)
