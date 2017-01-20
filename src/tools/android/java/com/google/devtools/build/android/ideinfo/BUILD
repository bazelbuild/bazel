filegroup(
    name = "embedded_tools",
    srcs = [
        "BUILD.tools",
        "libjar_filter_lib.jar",
        "libpackage_parser_lib.jar",
    ],
    visibility = ["//src:__pkg__"],
)

IDEINFO_DEPS = [
  "//src/main/java/com/google/devtools/common/options",
  "//src/main/protobuf:package_manifest_java_proto",
  "//src/tools/android/java/com/google/devtools/build/android:android_builder_lib",
  "//third_party:guava",
  "//third_party:jsr305",
  "//third_party/protobuf",
]

java_binary(
    name = "JarFilter",
    main_class = "com.google.devtools.build.android.ideinfo.JarFilter",
    visibility = ["//visibility:public"],
    runtime_deps = [":jar_filter_lib"],
)

java_library(
    name = "jar_filter_lib",
    srcs = ["JarFilter.java"],
    visibility = [
        "//devtools/blaze/integration:__pkg__",
        "//src/test/java/com/google/devtools/build/android/ideinfo:__pkg__",
    ],
    deps = IDEINFO_DEPS,
)

java_binary(
    name = "PackageParser",
    main_class = "com.google.devtools.build.android.ideinfo.PackageParser",
    visibility = ["//visibility:public"],
    runtime_deps = [":package_parser_lib"],
)

java_library(
    name = "package_parser_lib",
    srcs = [
        "ArtifactLocationConverter.java",
        "ArtifactLocationListConverter.java",
        "PackageParser.java",
        "PackageParserIoProvider.java",
    ],
    visibility = [
        "//devtools/blaze/integration:__pkg__",
        "//src/test/java/com/google/devtools/build/android/ideinfo:__pkg__",
    ],
    deps = IDEINFO_DEPS,
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/tools/android/java/com/google/devtools/build/android:__pkg__"],
)
