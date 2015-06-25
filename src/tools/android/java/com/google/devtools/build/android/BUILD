# Actions for Android rules.

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

java_binary(
    name = "AndroidResourceProcessingAction",
    main_class = "com.google.devtools.build.android.AndroidResourceProcessingAction",
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
        "//src/main/java:options",
        "//third_party:android_common",
        "//third_party:apache_commons_compress",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)
