load("//:distdir_deps.bzl", "gen_workspace_stanza")

package(default_testonly = True)

filegroup(
    name = "srcs",
    testonly = False,
    srcs = glob(["**"]),
    visibility = ["//src/test/shell/bazel:__pkg__"],
)

filegroup(
    name = "git-repos",
    srcs = [
        "outer-planets-repo.tar.gz",
        "pluto-repo.tar.gz",
        "refetch-repo.tar.gz",
        "strip-prefix-repo.tar.gz",
    ],
    visibility = ["//src/test/shell/bazel:__pkg__"],
)

filegroup(
    name = "embedded_tools_deps_test_data",
    srcs = [
        "//src/test/shell/bazel/testdata:embedded_tools_srcs_deps",
    ],
    visibility = ["//src/test/shell/bazel:__pkg__"],
)

filegroup(
    name = "jdk_http_archives_filegroup",
    srcs = [
        "//src/test/shell/bazel/testdata:jdk_http_archives",
    ],
    visibility = ["//src/test/shell/bazel:__subpackages__"],
)

gen_workspace_stanza(
    name = "workspace_with_rules_java",
    out = "jdk_http_archives",
    repo_clause = """
  name = "{repo}",
  sha256 = "{sha256}",
  strip_prefix = {strip_prefix},
  urls = {urls},
""",
    repos = [
        "remote_java_tools_test",
        "remote_java_tools_test_linux",
        "remote_java_tools_test_windows",
        "remote_java_tools_test_darwin_x86_64",
        "remote_java_tools_test_darwin_arm64",
        "openjdk11_linux_archive",
        "openjdk11_linux_s390x_archive",
        "openjdk11_darwin_archive",
        "openjdk11_darwin_aarch64_archive",
        "openjdk11_windows_archive",
        "openjdk11_windows_arm64_archive",
    ] + [
        "openjdk%s_%s_archive" % (version, os)
        for version in ("17", "20")
        for os in ("linux", "darwin", "darwin_aarch64", "windows") + (("linux_s390x", "windows_arm64") if version != "20" else ())
    ],
    template = "jdk_http_archives.tmpl",
    visibility = ["//:__pkg__"],
)

exports_files(["zstd_test_archive.tar.zst"])
