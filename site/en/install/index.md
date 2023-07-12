Project: /_project.yaml
Book: /_book.yaml

# Installing Bazel

{% dynamic setvar source_file "site/en/install/index.md" %}
{% include "_buttons.html" %}

This page describes the various platforms supported by Bazel and links
to the packages for more details.

[Bazelisk](/install/bazelisk) is the recommended way to install Bazel on [Ubuntu Linux](/install/ubuntu), [macOS](/install/os-x), and [Windows](/install/windows).

## Community-supported packages {:#community-supported-packages}

Bazel community members maintain these packages. The Bazel team doesn't
officially support them. Contact the package maintainers for support.

*   [Arch Linux][arch]{: .external}
*   [Fedora 25, 26, 27, 28, and CentOS 7](/install/redhat)
*   [CentOS 6](https://github.com/sub-mod/bazel-builds){: .external}
*   [Debian](https://qa.debian.org/developer.php?email=team%2Bbazel%40tracker.debian.org){: .external}
*   [FreeBSD](https://www.freshports.org/devel/bazel){: .external}
*   [Gentoo](https://packages.gentoo.org/packages/dev-util/bazel){: .external}
*   [Linuxbrew](https://github.com/Linuxbrew/homebrew-core/blob/master/Formula/bazel.rb){: .external}
*   [Nixpkgs](https://github.com/NixOS/nixpkgs/blob/master/pkgs/development/tools/build-managers/bazel){: .external}
*   [openSUSE](/install/suse)
*   [Parabola](https://www.parabola.nu/packages/?q=bazel){: .external}
*   [Scoop](https://github.com/scoopinstaller/scoop-main/blob/master/bucket/bazel.json){: .external}
*   [Raspberry Pi](https://github.com/koenvervloesem/bazel-on-arm/blob/master/README.md){: .external}

## Community-supported architectures {:#community-supported-architectures}

*   [ppc64el](https://ftp2.osuosl.org/pub/ppc64el/bazel/){: .external}

For other platforms, you can try to [compile from source](/install/compile-source).

[arch]: https://archlinux.org/packages/extra/x86_64/bazel/
