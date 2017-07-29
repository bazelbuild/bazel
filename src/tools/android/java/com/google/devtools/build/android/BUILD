# Actions for Android rules.

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "embedded_tools",
    srcs = [
        "BUILD.tools",
        "classes_deploy.jar",
        "//src/tools/android/java/com/google/devtools/build/android/desugar:embedded_tools",
        "//src/tools/android/java/com/google/devtools/build/android/incrementaldeployment:embedded_tools",
        "//src/tools/android/java/com/google/devtools/build/android/junctions:embedded_tools",
        "//src/tools/android/java/com/google/devtools/build/android/proto:srcs",
    ],
)

java_binary(
    name = "classes",
    main_class = "does.not.exist",
    runtime_deps = [":android_builder_lib"],
)

# Used by //src/test/shell/bazel/android:android_integration_test
java_binary(
    name = "ResourceProcessorBusyBox",
    main_class = "com.google.devtools.build.android.ResourceProcessorBusyBox",
    runtime_deps = [
        ":android_builder_lib",
    ],
)

java_library(
    name = "android_builder_lib",
    srcs = glob([
        "*.java",
        "xml/*.java",
    ]),
    deps = [
        "//src/main/java/com/google/devtools/common/options",
        "//src/tools/android/java/com/google/devtools/build/android/junctions",
        "//src/tools/android/java/com/google/devtools/build/android/proto:serialize_format_java_pb",
        "//src/tools/android/java/com/google/devtools/build/android/resources",
        "//third_party:android_common_25_0_0",
        "//third_party:asm",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/java/android_databinding:exec",
        "//third_party/java/aosp_gradle_core",
        "//third_party/protobuf:protobuf_java",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/tools/android/java/com/google/devtools/build/android/desugar:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/dexer:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/idlclass:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/incrementaldeployment:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/junctions:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/proto:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/resources:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/ziputils:srcs",
    ],
    visibility = ["//src:__pkg__"],
)
