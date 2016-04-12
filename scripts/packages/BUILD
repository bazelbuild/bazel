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

genrule(
    name = "rename-bazel-bin",
    srcs = ["//src:bazel"],
    outs = ["bazel-real"],
    cmd = "cp $< $@",
)

genrule(
    name = "rename-bazel-bin-jdk7",
    srcs = ["//src:bazel_jdk7"],
    outs = ["jdk7/bazel-real"],
    cmd = "cp $< $@",
)

genrule(
    name = "rename-bazel-sh",
    srcs = ["bazel.sh"],
    outs = ["bazel"],
    cmd = "cp $< $@",
)

load(":self_extract_binary.bzl", "self_extract_binary")

[
    self_extract_binary(
        name = "install%s.sh" % flavour,
        empty_files = ["tools/defaults/BUILD"],
        flatten_resources = [
            ":bazel-real" + flavour,
            ":bazel",
            "//scripts:bash_completion",
        ],
        launcher = ":launcher_bin.sh",
    )
    for flavour in [
        "",
        "-jdk7",
    ]
]

load("//tools/build_defs/pkg:pkg.bzl", "pkg_tar", "pkg_deb")

pkg_tar(
    name = "bazel-real-bin",
    files = [
        ":bazel-real",
    ],
    mode = "0755",
    package_dir = "/usr/bin",
    strip_prefix = ".",
)

pkg_tar(
    name = "bazel-real-bin-jdk7",
    files = [
        ":bazel-real-jdk7",
    ],
    mode = "0755",
    package_dir = "/usr/bin",
    strip_prefix = "scripts/packages/jdk7/",
)

pkg_tar(
    name = "bazel-bin",
    files = [
        ":bazel",
    ],
    mode = "0755",
    package_dir = "/usr/bin",
    strip_prefix = ".",
)

pkg_tar(
    name = "bazel-rc",
    files = ["debian/bazel.bazelrc"],
    mode = "0644",
    package_dir = "/etc",
    strip_prefix = "debian",
)

genrule(
    name = "rename-bazel-completion",
    srcs = ["//scripts:bash_completion"],
    outs = ["etc/bash_completion.d/bazel"],
    cmd = "cp $< $@",
)

pkg_tar(
    name = "bazel-completion",
    files = [":etc/bash_completion.d/bazel"],
    mode = "0644",
    strip_prefix = ".",
)

[
    pkg_tar(
        name = "debian-data" + flavour,
        extension = "tar.gz",
        deps = [
            ":bazel-real-bin" + flavour,
            ":bazel-bin",
            ":bazel-completion",
            ":bazel-rc",
        ],
    )
    for flavour in [
        "",
        "-jdk7",
    ]
]

[
    genrule(
        name = "embedded_label" + flavour,
        srcs = [":debian-data"],
        outs = [
            "version%s.txt" % flavour,
        ],
        cmd =
            "find . -name stable-status.txt -exec cat '{}' ';'" +
            " | grep EMBED_LABEL | cut -d ' ' -f 2-" +
            # Debian package manager does not order our rcs correctly, insert a tilde
            # Non RC version number are not matched by the regex, example translations:
            # 0.1.2         0.1.2
            # 0.1.2rc2      0.1.2~rc2
            # 0.1.2rc2-jdk7 0.1.2~rc2-jdk7
            " | sed -E 's/^([0-9\\.]+)([^0-9\\.].*)$$/\\1~\\2/'" +
            " > $@\n",
        stamp = 1,
    )
    for flavour in [
        "",
        "-jdk7",
    ]
]

[
    pkg_deb(
        name = "bazel-debian" + flavour,
        architecture = "amd64",
        built_using = "bazel (HEAD)",
        data = ":debian-data" + flavour,
        depends = [
            "java7-jdk | java7-sdk" if (flavour == "-jdk7") else "java8-jdk | java8-sdk",
            "pkg-config",
            "zip",
            "g++",
            "zlib1g-dev",
            "unzip",
            "bash-completion",
        ],
        description_file = "debian/description",
        homepage = "http://bazel.io",
        maintainer = "The Bazel Authors <bazel-dev@googlegroups.com>",
        package = "bazel",
        version_file = ":version%s.txt" % flavour,
    )
    for flavour in [
        "",
        "-jdk7",
    ]
]
