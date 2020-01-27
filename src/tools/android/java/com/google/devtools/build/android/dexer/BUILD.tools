package(default_visibility = ["//visibility:public"])

java_library(
    name = "dexer",
    srcs = glob(["*.java"]),
    plugins = ["auto_value_plugin"],
    deps = [
        "//external:android/dx_jar_import",
        "//src/tools/android/java/com/google/devtools/build/android:all_android_tools",
    ],
)

java_plugin(
    name = "auto_value_plugin",
    processor_class = "com.google.auto.value.processor.AutoValueProcessor",
    deps = ["//src/tools/android/java/com/google/devtools/build/android:all_android_tools"],
)

java_binary(
    name = "DexBuilder",
    main_class = "com.google.devtools.build.android.dexer.DexBuilder",
    visibility = ["//tools/android:__subpackages__"],
    runtime_deps = [":dexer"],
)

java_binary(
    name = "DexFileMerger",
    main_class = "com.google.devtools.build.android.dexer.DexFileMerger",
    visibility = ["//tools/android:__subpackages__"],
    runtime_deps = [":dexer"],
)

java_binary(
    name = "DexFileSplitter",
    main_class = "com.google.devtools.build.android.dexer.DexFileSplitter",
    visibility = ["//tools/android:__subpackages__"],
    runtime_deps = [":dexer"],
)
