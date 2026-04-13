load("@rules_java//java:defs.bzl", "java_binary")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_binary(
    name = "BazelPackageLoaderTester",
    testonly = True,
    srcs = ["BazelPackageLoaderTester.java"],
    jvm_flags = select({
        # For MacService machines which only have IPv6 addresses.
        "//src/conditions:darwin": ["-Djava.net.preferIPv6Addresses=true"],
        "//conditions:default": [],
    }),
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/skyframe/packages:BazelPackageLoader",
        "//src/main/java/com/google/devtools/build/lib/skyframe/packages:PackageLoader",
        "//src/main/java/com/google/devtools/build/lib/unix:native_posix_files_service_impl",
        "//src/main/java/com/google/devtools/build/lib/unix:unix_file_system",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party:guava",
    ],
)
