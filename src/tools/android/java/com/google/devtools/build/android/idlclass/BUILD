filegroup(
    name = "srcs",
    # include this BUILD file for the default android tools repository
    srcs = glob(["**"]),
    visibility = ["//src:__pkg__"],
)

java_binary(
    name = "IdlClass",
    main_class = "com.google.devtools.build.android.idlclass.IdlClass",
    visibility = ["//visibility:public"],
    runtime_deps = [":idlclass_lib"],
)

java_library(
    name = "idlclass_lib",
    srcs = glob(["*.java"]),
    visibility = [
        "//devtools/blaze/integration:__pkg__",
        "//src/test/java/com/google/devtools/build/android/idlclass:__pkg__",
    ],
    deps = [
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar/jarhelper",
        "//src/main/java:options",
        "//src/main/protobuf:proto_java_compilation",
        "//src/tools/android/java/com/google/devtools/build/android:android_builder_lib",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)
