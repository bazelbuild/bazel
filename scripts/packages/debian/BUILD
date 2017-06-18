filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//scripts:__subpackages__"],
)

load("//tools/build_defs/pkg:pkg.bzl", "pkg_tar", "pkg_deb")

pkg_tar(
    name = "bazel-bin",
    files = [
        "//scripts/packages:without-jdk/bazel",
        "//scripts/packages:without-jdk/bazel-real",
    ],
    mode = "0755",
    package_dir = "/usr/bin",
    strip_prefix = "/scripts/packages/without-jdk",
)

pkg_tar(
    name = "bazel-rc",
    files = ["//scripts/packages:bazel.bazelrc"],
    mode = "0644",
    package_dir = "/etc",
    strip_prefix = "/scripts/packages",
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

pkg_tar(
    name = "debian-data",
    extension = "tar.gz",
    deps = [
        ":bazel-bin",
        ":bazel-completion",
        ":bazel-rc",
    ],
)

genrule(
    name = "debian_version",
    srcs = ["//scripts/packages:version.txt"],
    outs = [
        "version.txt",
    ],
    cmd =
        "cat $(location //scripts/packages:version.txt) " +
        # Debian package manager does not order our rcs correctly, insert a tilde
        # Non RC version number are not matched by the regex, example translations:
        # 0.1.2         0.1.2
        # 0.1.2rc2      0.1.2~rc2
        # 0.1.2rc2-jdk7 0.1.2~rc2-jdk7
        " | sed -E 's/^([0-9\\.]+)([^0-9\\.].*)$$/\\1~\\2/'" +
        " > $@\n",
)

pkg_deb(
    name = "bazel-debian",
    architecture = "amd64",
    built_using = "bazel (HEAD)",
    conflicts = [
        "openjdk-9-jdk",
    ],
    data = ":debian-data",
    depends = select({
        "//tools/jdk:jdk7": [
            "java7-jdk | java7-sdk | oracle-java7-installer",
        ],
        "//conditions:default": [
            "google-jdk | java8-jdk | java8-sdk | oracle-java8-installer",
        ],
    }) + [
        "g++",
        "zlib1g-dev",
        "bash-completion",
    ],
    description_file = "description",
    homepage = "http://bazel.build",
    maintainer = "The Bazel Authors <bazel-dev@googlegroups.com>",
    package = "bazel",
    version_file = ":version.txt",
    visibility = ["//scripts/packages:__pkg__"],
)

filegroup(
    name = "debian-files",
    srcs = [
        "compat",
        "control",
        "copyright",
        "rules",
    ],
)

genrule(
    name = "generate-changelog-file",
    srcs = [
        "convert_changelog.py",
        "//:changelog-file",
        "//:bazel-srcs",  # Force a rebuild on source change
    ],
    outs = ["changelog"],
    cmd = "python $(location convert_changelog.py) bazel-out/volatile-status.txt $(location //:changelog-file) $(location changelog)",
    stamp = 1,
)

genrule(
    name = "bazel-debian-src",
    srcs = [
        "//:bazel-srcs",
        ":debian-files",
        ":changelog",
    ],
    outs = [
        "bazel.dsc",
        "bazel.tar.gz",
    ],
    cmd = """
        mkdir -p bazel/debian
        tar -xf $(location //:bazel-srcs) -C ./bazel
        for f in $(locations :debian-files); do
          cp $$f ./bazel/debian/
        done
        cp $(location :changelog) ./bazel/debian
        dpkg-source -b ./bazel
        cp ./bazel_*.dsc $(location bazel.dsc)
        cp ./bazel_*.tar.gz $(location bazel.tar.gz)
        """,
    visibility = ["//scripts/packages:__pkg__"],
)
