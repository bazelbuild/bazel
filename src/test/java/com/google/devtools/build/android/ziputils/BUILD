java_test(
    name = "ziputils-tests",
    srcs = glob(["*.java"]),
    args = ["com.google.devtools.build.android.ziputils.AllTests"],
    tags = ["ziputils"],
    deps = [
        "//src/test/java/com/google/devtools/build/lib:testutil",
        "//src/tools/android/java/com/google/devtools/build/android/ziputils:splitter_lib",
        "//src/tools/android/java/com/google/devtools/build/android/ziputils:ziputils_lib",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)
