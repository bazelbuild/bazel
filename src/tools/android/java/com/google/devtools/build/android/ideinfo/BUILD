filegroup(
    name = "embedded_tools",
    srcs = [
        "BUILD.tools",
        "classes_deploy.jar",
    ],
    visibility = ["//src:__pkg__"],
)

java_binary(
    name = "classes",
    main_class = "does.not.exist",
    runtime_deps = [":package_parser_lib"],
)

java_binary(
    name = "PackageParser",
    main_class = "com.google.devtools.build.android.ideinfo.PackageParser",
    visibility = ["//visibility:public"],
    runtime_deps = [":package_parser_lib"],
)

java_library(
    name = "package_parser_lib",
    srcs = glob(["*.java"]),
    visibility = [
        "//devtools/blaze/integration:__pkg__",
        "//src/test/java/com/google/devtools/build/android/ideinfo:__pkg__",
    ],
    deps = [
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:package_manifest_java_proto",
        "//src/tools/android/java/com/google/devtools/build/android:android_builder_lib",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:protobuf",
    ],
)
