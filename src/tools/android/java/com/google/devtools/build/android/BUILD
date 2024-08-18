# Actions for Android rules.

load("@rules_java//java:defs.bzl", "java_binary", "java_library")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "embedded_tools",
    srcs = [
        "BUILD.tools",
        "//src/tools/android/java/com/google/devtools/build/android/desugar:embedded_tools",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/scan:embedded_tools",
        "//src/tools/android/java/com/google/devtools/build/android/dexer:embedded_tools",
        "//src/tools/android/java/com/google/devtools/build/android/idlclass:embedded_tools",
        "//src/tools/android/java/com/google/devtools/build/android/junctions:embedded_tools",
        "//src/tools/android/java/com/google/devtools/build/android/proto:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/r8:embedded_tools",
        "//src/tools/android/java/com/google/devtools/build/android/ziputils:embedded_tools",
    ],
)

java_binary(
    name = "all_android_tools",
    create_executable = 0,
    runtime_deps = [
        ":android_builder_lib",
        "//src/main/java/com/google/devtools/build/lib/worker:work_request_handlers",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/dependencies",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/io",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/langmodel",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/scan",
        "//src/tools/android/java/com/google/devtools/build/android/dexer:dexerdeps",
        "//src/tools/android/java/com/google/devtools/build/android/idlclass:idlclass_lib",
        "//src/tools/android/java/com/google/devtools/build/android/r8:dexerdeps",
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

java_binary(
    name = "ZipFilterAction",
    # Memory consumption of SingleJar is about 250 bytes per entry in the output file. Unfortunately,
    # the JVM tends to kill the process with an OOM long before we're at the limit. In the most
    # recent example, 400 MB of memory was enough for about 500,000 entries.
    jvm_flags = ["-Xmx1600m"],
    main_class = "com.google.devtools.build.android.ZipFilterAction",
    visibility = ["//visibility:private"],
    runtime_deps = [":android_builder_lib"],
)

java_library(
    name = "android_builder_lib",
    srcs = glob(
        [
            "*.java",
            "xml/*.java",
            "aapt2/*.java",
        ],
        exclude = ["DependencyInfo.java"],
    ),
    javacopts = [
        "--add-exports=java.xml/com.sun.org.apache.xerces.internal.dom=ALL-UNNAMED",
    ],
    exports = [
        # Separated out to avoid a dependency cycle with resources/, but it logically belongs here.
        ":dependency_info",
    ],
    deps = [
        ":android_common_30_1_3",
        ":dependency_info",
        "//src/java_tools/singlejar/java/com/google/devtools/build/singlejar:libSingleJar",
        "//src/java_tools/singlejar/java/com/google/devtools/build/zip",
        "//src/main/java/com/google/devtools/build/lib/worker",
        "//src/main/java/com/google/devtools/build/lib/worker:work_request_handlers",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:worker_protocol_java_proto",
        "//src/tools/android/java/com/google/devtools/build/android/junctions",
        "//src/tools/android/java/com/google/devtools/build/android/proto:resources_java_proto",
        "//src/tools/android/java/com/google/devtools/build/android/proto:serialize_format_java_pb",
        "//src/tools/android/java/com/google/devtools/build/android/resources",
        "//src/tools/android/java/com/google/devtools/build/android/ziputils:ziputils_lib",
        "//third_party:asm",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/java/android_databinding:exec",
        "//third_party/java/aosp_gradle_core:aosp_gradle_core_30_1_3",
        "//third_party/java/jcommander",
        "//third_party/protobuf:protobuf_java",
    ],
)

java_library(
    name = "dependency_info",
    srcs = ["DependencyInfo.java"],
    visibility = [":__subpackages__"],
    deps = [
        "//third_party:auto_value",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/tools/android/java/com/google/devtools/build/android/desugar:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/dexer:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/idlclass:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/junctions:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/proto:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/r8:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/resources:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/ziputils:srcs",
    ],
    visibility = ["//src:__pkg__"],
)

java_import(
    name = "android_common_30_1_3",
    jars = [
        "@maven_android//:com_android_signflinger_file",
        "@maven_android//:com_android_tools_analytics_library_protos_file",
        "@maven_android//:com_android_tools_analytics_library_shared_file",
        "@maven_android//:com_android_tools_analytics_library_tracker_file",
        "@maven_android//:com_android_tools_annotations_file",
        "@maven_android//:com_android_tools_build_apksig_file",
        "@maven_android//:com_android_tools_build_apkzlib_file",
        "@maven_android//:com_android_tools_build_builder_file",
        "@maven_android//:com_android_tools_build_builder_model_file",
        "@maven_android//:com_android_tools_build_manifest_merger_file",
        "@maven_android//:com_android_tools_common_file",
        "@maven_android//:com_android_tools_ddms_ddmlib_file",
        "@maven_android//:com_android_tools_dvlib_file",
        "@maven_android//:com_android_tools_layoutlib_layoutlib_api_file",
        "@maven_android//:com_android_tools_repository_file",
        "@maven_android//:com_android_tools_sdk_common_file",
        "@maven_android//:com_android_tools_sdklib_file",
        "@maven_android//:com_android_zipflinger_file",
    ],
    runtime_deps = [
        "@maven_android//:org_ow2_asm_asm",
        "@maven_android//:org_ow2_asm_asm_commons",
        "@maven_android//:org_ow2_asm_asm_tree",
        "@maven_android//:org_ow2_asm_asm_util",
    ],
    deps = [
        "@maven_android//:org_glassfish_jaxb_jaxb_runtime",
        "@maven_android//:org_jetbrains_kotlin_kotlin_stdlib_jdk7",
    ],
)
