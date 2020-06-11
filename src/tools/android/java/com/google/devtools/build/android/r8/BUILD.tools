package(default_visibility = ["//visibility:public"])

java_library(
    name = "r8",
    srcs = glob(["*.java"]),
    deps = [
        "//external:android/d8_jar_import",
    ],
)

java_binary(
    name = "DexBuilder",
    main_class = "com.google.devtools.build.android.r8.DexBuilder",
    visibility = ["//tools/android:__subpackages__"],
    runtime_deps = [":r8"],
)

java_binary(
    name = "DexFileMerger",
    main_class = "com.google.devtools.build.android.r8.DexFileMerger",
    visibility = ["//tools/android:__subpackages__"],
    runtime_deps = [":r8"],
)
