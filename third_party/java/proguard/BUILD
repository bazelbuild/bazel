licenses(["restricted"])  # GNU GPL v2

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//third_party:__pkg__"],
)

filegroup(
    name = "embedded_tools",
    srcs = [
        "BUILD.tools",
        "proguard5.3.3/docs/GPL.html",
        "proguard5.3.3/lib/proguard.jar",
    ],
    visibility = ["//src:__pkg__"],
)

java_import(
    name = "proguard_import",
    jars = ["proguard5.3.3/lib/proguard.jar"],
)

java_binary(
    name = "proguard",
    main_class = "proguard.ProGuard",
    runtime_deps = [":proguard_import"],
)

# To be bundled into remote Java tools
filegroup(
    name = "proguard_deploy_with_license",
    srcs = [
        "proguard5.3.3/docs/GPL.html",
        ":proguard_deploy.jar",
    ],
    visibility = ["//src:__pkg__"],
)
