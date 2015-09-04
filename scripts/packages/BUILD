sh_binary(
    name = "package-info-generator",
    srcs = ["package_info_generator.sh"],
)

genrule(
    name = "generate-package-info",
    outs = ["README.md"],
    cmd = "$(location :package-info-generator) $$(find . -name '*status*.txt') >$@",
    stamp = 1,
    tools = [":package-info-generator"],
)

genrule(
    name = "generate-launcher",
    srcs = [
        "template_bin.sh",
        ":README.md",
    ],
    outs = ["launcher_bin.sh"],
    cmd = """
        release_info="$$(cat $(location :README.md))"
        template="$$(cat $(location template_bin.sh))"
        echo "$${template//%release_info%/$${release_info}}" >$@
        """,
)

load("self_extract_binary", "self_extract_binary")

self_extract_binary(
    name = "install.sh",
    empty_files = ["tools/defaults/BUILD"],
    flatten_resources = [
        "//src:bazel",
        "//scripts:bash_completion",
    ],
    launcher = ":launcher_bin.sh",
    resources = [
        "//third_party:srcs",
        "//third_party/java/jdk/langtools:srcs",
        "//tools:package-srcs",
    ],
)
