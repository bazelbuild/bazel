# Description:
#   Part of the Bazel DocGen pipeline that updates the table-of-contents of the release documentation
package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//:__pkg__"],
)

java_binary(
    name = "toc_updater",
    srcs = [
        "TableOfContentsOptions.java",
        "TableOfContentsUpdater.java",
    ],
    main_class = "com.google.devtools.build.docgen.release.TableOfContentsUpdater",
    visibility = ["//scripts/docs:__pkg__"],
    deps = [
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:flogger",
        "@maven//:org_yaml_snakeyaml",
    ],
)
