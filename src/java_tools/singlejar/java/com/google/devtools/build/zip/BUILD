load("@rules_java//java:defs.bzl", "java_binary", "java_library")

# Description:
#   Zip provides a library for reading and writing zip files, allowing more
#   advanced manipulation than the JDK equivalents by providing detailed zip
#   entry data and raw file access.
package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src/java_tools/singlejar:singlejar_package_group"],
)

filegroup(
    name = "srcs",
    srcs = glob(["*.java"]) + ["BUILD"],
    visibility = ["//src:__subpackages__"],
)

# Necessary for the bootstrap compilation.
filegroup(
    name = "java-srcs",
    srcs = glob(["*.java"]),
)

java_library(
    name = "zip",
    srcs = glob(["*.java"]),
    visibility = ["//visibility:public"],
    deps = ["//third_party:jsr305"],
)

# Adjusts zip offsets in self-extracting archive (like zip -qA, but out-of-place).
java_binary(
    name = "adjust_sfx",
    main_class = "com.google.devtools.build.zip.AdjustSfx",
    visibility = ["//src:__pkg__"],
    runtime_deps = [":zip"],
)
