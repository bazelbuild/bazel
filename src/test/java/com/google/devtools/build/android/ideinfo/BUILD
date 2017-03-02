filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/test/java/com/google/devtools/build/android:__pkg__"],
)

java_test(
    name = "JarFilterTest",
    size = "small",
    srcs = ["JarFilterTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:package_manifest_java_proto",
        "//src/tools/android/java/com/google/devtools/build/android/ideinfo:jar_filter_lib",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
        "//third_party/protobuf:protobuf_java",
    ],
)

java_test(
    name = "PackageParserTest",
    size = "small",
    srcs = ["PackageParserTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:package_manifest_java_proto",
        "//src/tools/android/java/com/google/devtools/build/android/ideinfo:package_parser_lib",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
        "//third_party/protobuf:protobuf_java",
    ],
)

java_test(
    name = "ArtifactLocationConverterTest",
    size = "small",
    srcs = ["ArtifactLocationConverterTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:package_manifest_java_proto",
        "//src/tools/android/java/com/google/devtools/build/android/ideinfo:package_parser_lib",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
        "//third_party/protobuf:protobuf_java",
    ],
)
