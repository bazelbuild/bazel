# Actions for Android rules.

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "embedded_tools",
    srcs = [
        "BUILD.tools",
        "classes_deploy.jar",
    ],
)

java_binary(
    name = "classes",
    main_class = "does.not.exist",
    runtime_deps = [":android_builder_lib"],
)

java_binary(
    name = "AndroidResourceProcessingAction",
    main_class = "com.google.devtools.build.android.AndroidResourceProcessingAction",
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

java_binary(
    name = "AarGeneratorAction",
    main_class = "com.google.devtools.build.android.AarGeneratorAction",
    runtime_deps = [
        ":android_builder_lib",
    ],
)

java_library(
    name = "android_builder_lib",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:package_manifest_java_proto",
        "//third_party:android_common",
        "//third_party:apache_commons_compress",
        "//third_party:asm",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:protobuf",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/tools/android/java/com/google/devtools/build/android/ideinfo:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/idlclass:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/incrementaldeployment:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/ziputils:srcs",
    ],
    visibility = ["//src:__pkg__"],
)
