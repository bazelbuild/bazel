package(default_visibility = ["//scripts/packages:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//scripts/packages/debian:srcs",
        "//scripts/packages/fedora:srcs",
    ],
    visibility = [
        "//scripts:__pkg__",
        "//scripts/packages:__subpackages__",
    ],
)

filegroup(
    name = "packages",
    srcs = select({
        "//src/conditions:windows": [
            ":zip-bazel-exe",
            ":zip-bazel-exe_nojdk",
        ],
        "//src/conditions:freebsd": [],
        "//src/conditions:openbsd": [],
        "//src/conditions:darwin": [
            ":with-jdk/install.sh",
            ":without-jdk/install.sh",
            ":generate-package-info",
        ],
        "//src/conditions:darwin_x86_64": [
            ":with-jdk/install.sh",
            ":without-jdk/install.sh",
            ":generate-package-info",
        ],
        "//conditions:default": [
            ":with-jdk/install.sh",
            ":without-jdk/install.sh",
            ":generate-package-info",
            "//:bazel-distfile",
            "//scripts/packages/debian:bazel-debian",
            "//scripts/packages/debian:bazel-debian-src",
        ],
    }),
)

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

[genrule(
    name = "zip-bazel-exe" + suffix,
    srcs = ["//src:bazel-bin%s.exe" % suffix],
    outs = ["bazel%s.zip" % suffix],
    cmd = "\n".join([
        "tmpdir=$$(mktemp -d $${TMPDIR:-/tmp}/tmp.XXXXXXXX)",
        "trap \"rm -fr $${tmpdir}\" EXIT",
        "cp $(SRCS) $${tmpdir}/bazel.exe",
        "touch -t 198001010000.00 $${tmpdir}/bazel.exe",
        "zip -jq $@ $${tmpdir}/bazel.exe",
    ]),
) for suffix in [
    "",
    "_nojdk",
]]

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

genrule(
    name = "rename-bazel-bin",
    srcs = ["//src:bazel"],
    outs = ["with-jdk/bazel-real"],
    cmd = "mkdir -p $$(dirname $@); cp $< $@",
)

genrule(
    name = "rename-bazel-bin-without-jdk",
    srcs = ["//src:bazel_nojdk"],
    outs = ["without-jdk/bazel-real"],
    cmd = "mkdir -p $$(dirname $@); cp $< $@",
)

genrule(
    name = "bazel-sh-with-jdk",
    srcs = ["bazel.sh"],
    outs = ["with-jdk/bazel"],
    cmd = "mkdir -p $$(dirname $@); cp $< $@",
)

genrule(
    name = "bazel-sh-without-jdk",
    srcs = ["bazel.sh"],
    outs = ["without-jdk/bazel"],
    cmd = "mkdir -p $$(dirname $@); cp $< $@",
)

load(":self_extract_binary.bzl", "self_extract_binary")

[self_extract_binary(
    name = "%s/install.sh" % kind,
    flatten_resources = [
        ":%s/bazel-real" % kind,
        ":%s/bazel" % kind,
        "//scripts:bash_completion",
        "//scripts/zsh_completion:_bazel",
    ],
    launcher = ":launcher_bin.sh",
) for kind in ("with-jdk", "without-jdk")]

genrule(
    name = "embedded_label",
    srcs = [],
    outs = [
        "version.txt",
    ],
    cmd =
        "find -L . -name stable-status.txt -exec cat '{}' ';'" +
        " | grep EMBED_LABEL | cut -d ' ' -f 2-" +
        " > $@\n",
    stamp = 1,
)
