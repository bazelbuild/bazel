load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

exports_files(
    ["docs/bazel-user-manual.html"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//:__pkg__"],
)

filegroup(
    name = "jekyll-srcs",
    srcs = glob(
        ["**/*"],
        exclude = [
            "BUILD",
            "jekyll-tree.sh",
            "*.swp",
        ],
    ),
)

pkg_tar(
    name = "jekyll-files",
    files = [":jekyll-srcs"],
    strip_prefix = ".",
)

pkg_tar(
    name = "bootstrap-css",
    files = ["//third_party/css/bootstrap:bootstrap_css"],
    package_dir = "assets",
    strip_prefix = "/third_party/css/bootstrap",
)

pkg_tar(
    name = "bootstrap-images",
    files = ["//third_party/css/bootstrap:bootstrap_images"],
    package_dir = "assets",
    strip_prefix = "/third_party/css/bootstrap",
)

pkg_tar(
    name = "font-awesome-css",
    files = ["//third_party/css/font_awesome:font_awesome_css"],
    package_dir = "assets",
    strip_prefix = "/third_party/css/font_awesome",
)

pkg_tar(
    name = "font-awesome-font",
    files = ["//third_party/css/font_awesome:font_awesome_font"],
    package_dir = "assets",
    strip_prefix = "/third_party/css/font_awesome",
)

pkg_tar(
    name = "bootstrap-js",
    files = ["//third_party/javascript/bootstrap:bootstrap_js"],
    package_dir = "assets",
    strip_prefix = "/third_party/javascript/bootstrap",
)

pkg_tar(
    name = "jekyll-base",
    deps = [
        ":bootstrap-css",
        ":bootstrap-images",
        ":bootstrap-js",
        ":font-awesome-css",
        ":font-awesome-font",
        ":jekyll-files",
    ],
)

pkg_tar(
    name = "skylark-rule-docs",
    files = [
        "//tools/build_defs/docker:README.md",
        "//tools/build_defs/pkg:README.md",
    ],
    strip_prefix = "/tools/build_defs",
)

genrule(
    name = "jekyll-tree",
    srcs = [
        ":jekyll-base",
        ":skylark-rule-docs",
        "//src/main/java/com/google/devtools/build/lib:gen_buildencyclopedia",
        "//src/main/java/com/google/devtools/build/lib:gen_skylarklibrary",
    ],
    outs = ["jekyll-tree.tar"],
    cmd = ("$(location jekyll-tree.sh) $@ " +
           "$(location :jekyll-base) " +
           "$(location :skylark-rule-docs) " +
           "$(location //src/main/java/com/google/devtools/build/lib:gen_buildencyclopedia) " +
           "$(location //src/main/java/com/google/devtools/build/lib:gen_skylarklibrary)"),
    tools = [
        "jekyll-tree.sh",
    ],
)
