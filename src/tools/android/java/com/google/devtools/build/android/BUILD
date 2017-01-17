# Actions for Android rules.

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "embedded_tools",
    srcs = [
        "BUILD.tools",
        "classes_deploy.jar",
        "//src/tools/android/java/com/google/devtools/build/android/proto:srcs",
    ],
)

java_binary(
    name = "classes",
    main_class = "does.not.exist",
    runtime_deps = [":android_builder_lib"],
)

java_binary(
    name = "AarGeneratorAction",
    main_class = "com.google.devtools.build.android.AarGeneratorAction",
    runtime_deps = [
        ":android_builder_lib",
    ],
)

java_binary(
    name = "AndroidResourceMergingAction",
    main_class = "com.google.devtools.build.android.AndroidResourceMergingAction",
    runtime_deps = [
        ":android_builder_lib",
    ],
)

java_binary(
    name = "AndroidResourceParsingAction",
    main_class = "com.google.devtools.build.android.AndroidResourceParsingAction",
    runtime_deps = [
        ":android_builder_lib",
    ],
)

java_binary(
    name = "AndroidResourceProcessingAction",
    main_class = "com.google.devtools.build.android.AndroidResourceProcessingAction",
    runtime_deps = [
        ":android_builder_lib",
    ],
)

java_binary(
    name = "AndroidResourceValidatorAction",
    main_class = "com.google.devtools.build.android.AndroidResourceValidatorAction",
    runtime_deps = [
        ":android_builder_lib",
    ],
)

java_binary(
    name = "ManifestMergerAction",
    main_class = "com.google.devtools.build.android.ManifestMergerAction",
    runtime_deps = [
        ":android_builder_lib",
    ],
)

java_binary(
    name = "RClassGeneratorAction",
    main_class = "com.google.devtools.build.android.RClassGeneratorAction",
    runtime_deps = [
        ":android_builder_lib",
    ],
)

java_binary(
    name = "LibraryRClassGeneratorAction",
    main_class = "com.google.devtools.build.android.LibraryRClassGeneratorAction",
    runtime_deps = [
        ":android_builder_lib",
    ],
)

java_binary(
    name = "ResourceShrinkerAction",
    main_class = "com.google.devtools.build.android.ResourceShinkerAction",
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
        "//src/main/protobuf:package_manifest_java_proto",
        "//src/tools/android/java/com/google/devtools/build/android/proto:serialize_format_java_pb",
        "//src/tools/android/java/com/google/devtools/build/android/resources",
        "//third_party:android_common_25_0_0",
        "//third_party:asm",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/java/android_databinding:exec",
        "//third_party/java/aosp_gradle_core",
        "//third_party/protobuf",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/tools/android/java/com/google/devtools/build/android/dexer:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/ideinfo:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/idlclass:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/incrementaldeployment:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/proto:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/resources:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/ziputils:srcs",
    ],
    visibility = ["//src:__pkg__"],
)
