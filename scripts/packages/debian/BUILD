load("@rules_pkg//pkg:deb.bzl", "pkg_deb")
load("@rules_pkg//pkg:tar.bzl", "pkg_tar")

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//scripts:__subpackages__"],
)

pkg_tar(
    name = "bazel-bin",
    srcs = [
        "//scripts/packages:with-jdk/bazel",
        "//scripts/packages:with-jdk/bazel-real",
    ],
    mode = "0755",
    package_dir = "/usr/bin",
    strip_prefix = "/scripts/packages/with-jdk",
)

pkg_tar(
    name = "bazel-rc",
    srcs = ["//scripts/packages:bazel.bazelrc"],
    mode = "0644",
    package_dir = "/etc",
    strip_prefix = "/scripts/packages",
)

genrule(
    name = "rename-bash-completion",
    srcs = ["//scripts:bash_completion"],
    outs = ["etc/bash_completion.d/bazel"],
    cmd = "cp $< $@",
)

genrule(
    name = "rename-zsh-completion",
    srcs = ["//scripts/zsh_completion:_bazel"],
    outs = ["usr/share/zsh/vendor-completions/_bazel"],
    cmd = "cp $< $@",
)

pkg_tar(
    name = "bazel-completion",
    srcs = [
        ":etc/bash_completion.d/bazel",
        ":usr/share/zsh/vendor-completions/_bazel",
    ],
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
    built_using = "bazel",
    conffiles = [
        "/etc/bash_completion.d/bazel",
        "/etc/bazel.bazelrc",
        "/usr/share/zsh/vendor-completions/_bazel",
    ],
    data = ":debian-data",
    depends = [
        # Keep in sync with Depends section in ./control
        "g++",
        "zlib1g-dev",
        "unzip",
    ],
    description_file = "description",
    homepage = "https://bazel.build",
    maintainer = "The Bazel Authors <bazel-discuss@googlegroups.com>",
    package = "bazel",
    suggests = [
        # Keep in sync with Suggests section in ./control
        "google-jdk | java8-sdk-headless | java8-jdk | java8-sdk | oracle-java8-installer",
        "bash-completion",
    ],
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
        "generate_changelog.py",
        "//:bazel-srcs",  # Force a rebuild on source change
    ],
    outs = ["changelog"],
    cmd = "python3 $(location generate_changelog.py) bazel-out/volatile-status.txt $(location changelog)",
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
