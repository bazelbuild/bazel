load("@rules_java//java:defs.bzl", "java_binary", "java_library")

filegroup(
    name = "embedded_tools",
    srcs = ["BUILD.tools"],
    visibility = [
        "//src/tools/android/java/com/google/devtools/build/android:__pkg__",
    ],
)

java_binary(
    name = "IdlClass",
    main_class = "com.google.devtools.build.android.idlclass.IdlClass",
    runtime_deps = [":idlclass_lib"],
)

java_library(
    name = "idlclass_lib",
    srcs = glob(["*.java"]),
    visibility = [
        "//src/test/java/com/google/devtools/build/android/idlclass:__pkg__",
        "//src/tools/android/java/com/google/devtools/build/android:__pkg__",
    ],
    deps = [
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/jarhelper",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:java_compilation_java_proto",
        "//src/tools/android/java/com/google/devtools/build/android:android_builder_lib",
        "//src/tools/android/java/com/google/devtools/build/android:android_options_utils",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/java/jcommander",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = [
        "//src/tools/android/java/com/google/devtools/build/android:__pkg__",
    ],
)
