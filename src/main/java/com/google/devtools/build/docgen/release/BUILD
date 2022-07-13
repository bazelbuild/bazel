licenses(["notice"])

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
    main_class = "com.google.devtools.build.docgen.TableOfContentsUpdater",
    visibility = ["//scripts/docs:__pkg__"],
    deps = [
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:flogger",
        "@org_snakeyaml//:snakeyaml",
    ],
)
