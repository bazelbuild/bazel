load("@rules_java//java:defs.bzl", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["*"]),
    visibility = ["//src:__subpackages__"],
)

java_test(
    name = "AllTests",
    size = "medium",
    srcs = glob(["*.java"]),
    jvm_flags = [
        "-Djava.lang.Thread.allowVirtualThreads=true",
    ],
    shard_count = 10,
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/packages:label_printer",
        "//src/main/java/com/google/devtools/build/lib/pkgcache",
        "//src/main/java/com/google/devtools/build/lib/query2",
        "//src/main/java/com/google/devtools/build/lib/query2/common:QueryTransitivePackagePreloader",
        "//src/main/java/com/google/devtools/build/lib/query2/common:abstract-blaze-query-env",
        "//src/main/java/com/google/devtools/build/lib/query2/common:universe-scope",
        "//src/main/java/com/google/devtools/build/lib/query2/engine",
        "//src/main/java/com/google/devtools/build/lib/util:string_encoding",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/test/java/com/google/devtools/build/lib/packages:testutil",
        "//src/test/java/com/google/devtools/build/lib/query2/testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestSuite",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:mockito",
        "//third_party:truth",
    ],
)
