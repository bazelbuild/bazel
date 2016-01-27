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
    name = "rename-bazel-sh",
    srcs = ["bazel.sh"],
    outs = ["bazel"],
    cmd = "cp $< $@",
)

load("self_extract_binary", "self_extract_binary")

self_extract_binary(
    name = "install.sh",
    empty_files = ["tools/defaults/BUILD"],
    flatten_resources = [
        ":bazel-real",
        ":bazel",
        "//scripts:bash_completion",
    ],
    launcher = ":launcher_bin.sh",
    resources = [
        "//third_party:srcs",
        "//third_party/java/jdk/langtools:srcs",
        "//tools:package-srcs",
    ],
)

load("/tools/build_defs/pkg/pkg", "pkg_tar", "pkg_deb")

pkg_tar(
    name = "bazel-bin",
    files = [
        ":bazel",
        ":bazel-real",
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

pkg_tar(
    name = "bazel-tools",
    files = [
        "//third_party:srcs",
        "//third_party/java/jdk/langtools:srcs",
        "//tools:package-srcs",
    ],
    mode = "0644",
    modes = {f: "0755" for f in [
        # List made out of `find -type f -executable`
        "third_party/iossim/iossim",
        "third_party/ijar/test/zip_test.sh",
        "third_party/ijar/test/ijar_test.sh",
        "third_party/ijar/test/testenv.sh",
        "third_party/protobuf/protoc-osx-x86_32.exe",
        "third_party/protobuf/protoc-linux-x86_64.exe",
        "third_party/protobuf/protoc-osx-x86_64.exe",
        "third_party/protobuf/protoc-linux-x86_32.exe",
        "third_party/protobuf/protoc-windows-x86_32.exe",
        "third_party/protobuf/protoc-windows-x86_64.exe",
        "third_party/py/gflags/tests/flags_modules_for_testing/module_bar.py",
        "third_party/py/gflags/tests/flags_modules_for_testing/module_baz.py",
        "third_party/py/gflags/tests/flags_modules_for_testing/module_foo.py",
        "third_party/py/gflags/tests/gflags_helpxml_test.py",
        "third_party/py/gflags/tests/gflags_unittest.py",
        "third_party/py/gflags/tests/gflags_validators_test.py",
        "third_party/py/gflags/gflags2man.py",
        "third_party/py/gflags/setup.py",
        "third_party/py/gflags/debian/rules",
        "third_party/py/gflags/gflags_validators.py",
        "third_party/py/mock/setup.py",
        "tools/android/jack/fail.sh",
        "tools/android/shuffle_jars.sh",
        "tools/android/merge_dexzips.sh",
        "tools/android/idlclass.sh",
        "tools/android/aar_generator.sh",
        "tools/android/resources_processor.sh",
        "tools/j2objc/j2objc_wrapper.py",
        "tools/genrule/genrule-setup.sh",
        "tools/objc/j2objc_dead_code_pruner.py",
        "tools/python/2to3.sh",
        "tools/cpp/osx_gcc_wrapper.sh",
        "tools/test/test-setup.sh",
        "tools/jdk/ijar",
        "tools/build_defs/docker/testenv.sh",
        "tools/build_defs/docker/build_test.sh",
    ]},
    package_dir = "/usr/share/lib/bazel/tools",
    strip_prefix = "/",
)

pkg_tar(
    name = "debian-data",
    extension = "tar.gz",
    deps = [
        ":bazel-bin",
        ":bazel-completion",
        ":bazel-rc",
        ":bazel-tools",
    ],
)

genrule(
    name = "embedded_label",
    srcs = [":debian-data"],
    outs = [
        "version.txt",
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

config_setting(
    name = "jdk7",
    values = {"define": "JAVA_VERSION=1.7"},
)

pkg_deb(
    name = "bazel-debian",
    architecture = "amd64",
    built_using = "bazel (HEAD)",
    data = ":debian-data",
    depends = select({
        ":jdk7": [
            "java7-jdk",
        ],
        "//conditions:default": [
            "java8-jdk",
        ],
    }) + [
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
    version_file = ":version.txt",
)
