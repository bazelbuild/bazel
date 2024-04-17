# Description:
#   Collection of dex utilities used in the bazel android actions.

load("@rules_java//java:defs.bzl", "java_library")

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = [
        "//src/test/java/com/google/devtools/build/android/r8:__pkg__",
        "//src/tools/android/java/com/google/devtools/build/android:__pkg__",
    ],
)

filegroup(
    name = "embedded_tools",
    srcs = glob(
        [
            "*.java",
            "desugar/*.java",
        ],
        exclude = [
            "NoAndroidSdkStub.java",
        ],
    ) + ["BUILD.tools"],
    visibility = ["//visibility:public"],
)

java_library(
    name = "r8",
    srcs = glob(
        [
            "*.java",
            "desugar/*.java",
        ],
        exclude = [
            "NoAndroidSdkStub.java",
        ],
    ),
    visibility = ["//src/test/java/com/google/devtools/build/android/r8:__pkg__"],
    runtime_deps = [
        "//src/tools/android/java/com/google/devtools/build/android/desugar/dependencies",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/worker:work_request_handlers",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:asm",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
        "@android_gmaven_r8//jar",
    ],
)

java_library(
    name = "utils",
    srcs = ["R8Utils.java"],
)

java_library(
    name = "deps_collector_api",
    srcs = ["DependencyCollector.java"],
    visibility = [
        "//src/test/java/com/google/devtools/build/android/desugar:__subpackages__",
        "//src/tools/android/java/com/google/devtools/build/android:__subpackages__",
        "//src/tools/android/java/com/google/devtools/build/android/r8:__subpackages__",
    ],
    deps = [
        ":utils",
        "//third_party:jsr305",
    ],
)

# The DexFileMerger and DexBuilder are built in BUILD.tools which is built in
# a developers workspace, not the Bazel workspace. So we must bundle the
# dependencies of those binaries into the embedded tools.
java_library(
    name = "dexerdeps",
    visibility = [
        "//src/tools/android/java/com/google/devtools/build/android:__pkg__",
    ],
    runtime_deps = [
        # CompatDexBuilder depends on some worker utilities.
        "//src/main/java/com/google/devtools/build/lib/worker:work_request_handlers",
    ],
)

java_binary(
    name = "desugar",
    jvm_flags = [
        # b/71513487
        "-XX:+TieredCompilation",
        "-XX:TieredStopAtLevel=1",
        "-Xms8g",
        "-Xmx8g",
        # b/172508621
        "-Dcom.android.tools.r8.sortMethodsOnCfWriting",
        "-Dcom.android.tools.r8.allowAllDesugaredInput",
        "-Dcom.android.tools.r8.noCfMarkerForDesugaredCode",
        "-Dcom.android.tools.r8.lambdaClassFieldsNotFinal",
    ],
    main_class = "com.google.devtools.build.android.r8.Desugar",
    visibility = ["//src/test/java/com/google/devtools/build/android/r8:__pkg__"],
    runtime_deps = [
        ":r8",
    ],
)

java_binary(
    name = "core_library_desugar",
    jvm_flags = [
        # b/71513487
        "-XX:+TieredCompilation",
        "-XX:TieredStopAtLevel=1",
        # b/172508621
        "-Dcom.android.tools.r8.sortMethodsOnCfWriting",
        "-Dcom.android.tools.r8.disableL8AnnotationRemoval",
    ],
    main_class = "com.google.devtools.build.android.r8.CoreLibraryDesugar",
    visibility = [
        "//visibility:public",
    ],
    runtime_deps = [
        ":r8",
    ],
)
