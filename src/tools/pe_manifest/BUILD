load("@rules_java//java:defs.bzl", "java_binary")

package(default_applicable_licenses = ["//:license"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/tools/pe_manifest/java/com/google/devtools/build/pemanifest:srcs",
        "//src/tools/pe_manifest/javatests/com/google/devtools/build/pemanifest:srcs",
    ],
    visibility = ["//src:__pkg__"],
)

# Reads or replaces the application manifest of a Windows executable on any
# platform, e.g. to patch the embedded JDK's java.exe when cross-compiling.
java_binary(
    name = "pe_manifest",
    main_class = "com.google.devtools.build.pemanifest.PeManifest",
    visibility = [
        "//src:__pkg__",
        "//src/java_tools/buildjar/java/com/google/devtools/build/java/turbine:__pkg__",
    ],
    runtime_deps = ["//src/tools/pe_manifest/java/com/google/devtools/build/pemanifest"],
)
