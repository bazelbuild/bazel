package(default_visibility = ["//visibility:public"])

java_binary(
    name = "mapper",
    main_class = "com.google.devtools.build.android.ziputils.DexMapper",
    runtime_deps = [
        "//src/tools/android/java/com/google/devtools/build/android:all_android_tools",
    ],
)

java_binary(
    name = "reducer",
    main_class = "com.google.devtools.build.android.ziputils.DexReducer",
    runtime_deps = [
        "//src/tools/android/java/com/google/devtools/build/android:all_android_tools",
    ],
)
