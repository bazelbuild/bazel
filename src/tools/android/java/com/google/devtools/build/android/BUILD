# Actions for Android rules.

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "embedded_tools",
    srcs = [
        "BUILD.tools",
        ":all_android_tools_deploy.jar",
        "//src/tools/android/java/com/google/devtools/build/android/desugar:embedded_tools",
        "//src/tools/android/java/com/google/devtools/build/android/dexer:embedded_tools",
        "//src/tools/android/java/com/google/devtools/build/android/idlclass:embedded_tools",
        "//src/tools/android/java/com/google/devtools/build/android/incrementaldeployment:embedded_tools",
        "//src/tools/android/java/com/google/devtools/build/android/junctions:embedded_tools",
        "//src/tools/android/java/com/google/devtools/build/android/proto:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/ziputils:embedded_tools",
    ],
)

java_binary(
    name = "all_android_tools",
    create_executable = 0,
    runtime_deps = [
        ":android_builder_lib",
        "//src/tools/android/java/com/google/devtools/build/android/desugar",
        "//src/tools/android/java/com/google/devtools/build/android/dexer:dexerdeps",
        "//src/tools/android/java/com/google/devtools/build/android/idlclass:idlclass_lib",
        "//src/tools/android/java/com/google/devtools/build/android/ziputils:ziputils_lib",
    ],
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
        "aapt2/*.java",
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
