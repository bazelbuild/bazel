workspace(name = "io_bazel")

load("//tools/build_defs/repo:http.bzl", "http_archive", "http_jar")
load("//:distdir.bzl", "dist_http_archive", "dist_http_file", "distdir_tar")
load("//:distdir_deps.bzl", "DIST_DEPS")

# These can be used as values for the patch_cmds and patch_cmds_win attributes
# of http_archive, in order to export the WORKSPACE file from the BUILD or
# BUILD.bazel file. This is useful for cases like //src:test_repos, where we
# have to be able to trigger a fetch of a repo by depending on it, but we don't
# actually want to build anything (so we can't depend on a target inside that
# repo).
EXPORT_WORKSPACE_IN_BUILD_FILE = [
    "test -f BUILD && chmod u+w BUILD || true",
    "echo >> BUILD",
    "echo 'exports_files([\"WORKSPACE\"], visibility = [\"//visibility:public\"])' >> BUILD",
]

EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE = [
    "test -f BUILD.bazel && chmod u+w BUILD.bazel || true",
    "echo >> BUILD.bazel",
    "echo 'exports_files([\"WORKSPACE\"], visibility = [\"//visibility:public\"])' >> BUILD.bazel",
]

EXPORT_WORKSPACE_IN_BUILD_FILE_WIN = [
    "Add-Content -Path BUILD -Value \"`nexports_files([`\"WORKSPACE`\"], visibility = [`\"//visibility:public`\"])`n\" -Force",
]

EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE_WIN = [
    "Add-Content -Path BUILD.bazel -Value \"`nexports_files([`\"WORKSPACE`\"], visibility = [`\"//visibility:public`\"])`n\" -Force",
]

# Protobuf expects an //external:python_headers label which would contain the
# Python headers if fast Python protos is enabled. Since we are not using fast
# Python protos, bind python_headers to a dummy target.
bind(
    name = "python_headers",
    actual = "//:dummy",
)

# Protobuf code generation for GRPC requires three external labels:
# //external:grpc-java_plugin
# //external:grpc-jar
# //external:guava
bind(
    name = "grpc-java-plugin",
    actual = "//third_party/grpc-java:grpc-java-plugin",
)

bind(
    name = "grpc-jar",
    actual = "//third_party/grpc-java:grpc-jar",
)

bind(
    name = "guava",
    actual = "//third_party:guava",
)

# We must control the version of rules_license we use, so we load ours before
# any other repo can bring it in through their deps.
dist_http_archive(
    name = "rules_license",
)

# For src/test/shell/bazel:test_srcs
load("//src/test/shell/bazel:list_source_repository.bzl", "list_source_repository")

list_source_repository(name = "local_bazel_source_list")

# To run the Android integration tests in //src/test/shell/bazel/android:all or
# build the Android sample app in //examples/android/java/bazel:hello_world
#
#   1. Install an Android SDK and NDK from https://developer.android.com
#   2. Set the $ANDROID_HOME and $ANDROID_NDK_HOME environment variables
#   3. Uncomment the two lines below
#
# android_sdk_repository(name = "androidsdk")
# android_ndk_repository(name = "androidndk")

# In order to run //src/test/shell/bazel:maven_skylark_test, follow the
# instructions above for the Android integration tests and uncomment the
# following lines:
# load("//tools/build_defs/repo:maven_rules.bzl", "maven_dependency_plugin")
# maven_dependency_plugin()

# This allows rules written in Starlark to locate apple build tools.
bind(
    name = "xcrunwrapper",
    actual = "@bazel_tools//tools/objc:xcrunwrapper",
)

dist_http_archive(
    name = "com_google_protobuf",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_FILE_WIN,
)

# This is a mock version of bazelbuild/rules_python that contains only
# @rules_python//python:defs.bzl. It is used by protobuf.
# TODO(#9029): We could potentially replace this with the real @rules_python.
new_local_repository(
    name = "rules_python",
    build_file = "//third_party/rules_python:BUILD",
    path = "./third_party/rules_python",
    workspace_file = "//third_party/rules_python:rules_python.WORKSPACE",
)

local_repository(
    name = "googleapis",
    path = "./third_party/googleapis/",
)

local_repository(
    name = "remoteapis",
    path = "./third_party/remoteapis/",
)

dist_http_archive(
    name = "desugar_jdk_libs",
)

distdir_tar(
    name = "additional_distfiles",
    # Keep in sync with the archives fetched as part of building bazel.
    archives = [
        "android_tools_pkg-0.27.0.tar.gz",
        # for android_gmaven_r8
        "r8-3.3.28.jar",
    ],
    dirname = "derived/distdir",
    dist_deps = {dep: attrs for dep, attrs in DIST_DEPS.items() if "additional_distfiles" in attrs["used_in"]},
    sha256 = {
        "android_tools_pkg-0.27.0.tar.gz": "1afa4b7e13c82523c8b69e87f8d598c891ec7e2baa41d9e24e08becd723edb4d",
        "r8-3.3.28.jar": "8626ca32fb47aba7fddd2c897615e2e8ffcdb4d4b213572a2aefb3f838f01972",
    },
    urls = {
        "android_tools_pkg-0.27.0.tar.gz": [
            "https://mirror.bazel.build/bazel_android_tools/android_tools_pkg-0.27.0.tar.gz",
        ],
        "r8-3.3.28.jar": [
            "https://maven.google.com/com/android/tools/r8/3.3.28/r8-3.3.28.jar",
        ],
    },
)

# OpenJDK distributions used to create a version of Bazel bundled with the OpenJDK.
dist_http_file(
    name = "openjdk_linux_vanilla",
    downloaded_file_path = "zulu-linux-vanilla.tar.gz",
)

dist_http_file(
    name = "openjdk_linux_aarch64_vanilla",
    downloaded_file_path = "zulu-linux-aarch64-vanilla.tar.gz",
)

dist_http_file(
    name = "openjdk_linux_ppc64le_vanilla",
    downloaded_file_path = "adoptopenjdk-ppc64le-vanilla.tar.gz",
)

dist_http_file(
    name = "openjdk_linux_s390x_vanilla",
    downloaded_file_path = "adoptopenjdk-s390x-vanilla.tar.gz",
)

dist_http_file(
    name = "openjdk_macos_x86_64_vanilla",
    downloaded_file_path = "zulu-macos-vanilla.tar.gz",
)

dist_http_file(
    name = "openjdk_macos_aarch64_vanilla",
    downloaded_file_path = "zulu-macos-aarch64-vanilla.tar.gz",
)

dist_http_file(
    name = "openjdk_win_vanilla",
    downloaded_file_path = "zulu-win-vanilla.zip",
)

dist_http_file(
    name = "openjdk_win_arm64_vanilla",
    downloaded_file_path = "zulu-win-arm64.zip",
)

dist_http_archive(
    name = "bazelci_rules",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_FILE_WIN,
)

load("@bazelci_rules//:rbe_repo.bzl", "rbe_preconfig")

rbe_preconfig(
    name = "rbe_ubuntu1804_java11",
    toolchain = "ubuntu1804-bazel-java11",
)

http_archive(
    name = "com_google_googletest",
    sha256 = "81964fe578e9bd7c94dfdb09c8e4d6e6759e19967e397dbea48d1c10e45d0df2",
    strip_prefix = "googletest-release-1.12.1",
    urls = [
        "https://mirror.bazel.build/github.com/google/googletest/archive/refs/tags/release-1.12.1.tar.gz",
        "https://github.com/google/googletest/archive/refs/tags/release-1.12.1.tar.gz",
    ],
)

dist_http_archive(
    name = "bazel_skylib",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_FILE_WIN,
)

dist_http_archive(
    name = "zstd-jni",
    build_file = "//third_party:zstd-jni/zstd-jni.BUILD",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE_WIN,
    strip_prefix = "zstd-jni-1.5.2-3",
)

http_archive(
    name = "org_snakeyaml",
    build_file_content = """
java_library(
    name = "snakeyaml",
    srcs = glob(["src/main/**/*.java"]),
    visibility = [
        "@io_bazel//src/main/java/com/google/devtools/build/docgen/release:__pkg__",
        "@com_google_testparameterinjector//:__pkg__",
    ],
)
""",
    sha256 = "fd0e0cc6c5974fc8f08be3a15fb4a59954c7dd958b5b68186a803de6420b6e40",
    strip_prefix = "asomov-snakeyaml-b28f0b4d87c6",
    urls = ["https://mirror.bazel.build/bitbucket.org/asomov/snakeyaml/get/snakeyaml-1.28.tar.gz"],
)

http_archive(
    name = "com_google_testparameterinjector",
    build_file_content = """
java_library(
    name = "testparameterinjector",
    testonly = True,
    srcs = glob(["src/main/**/*.java"]),
    deps = [
      "@org_snakeyaml//:snakeyaml",
      "@//third_party:auto_value",
      "@//third_party:guava",
      "@//third_party:junit4",
      "@//third_party/protobuf:protobuf_java",
    ],
    visibility = ["//visibility:public"],
)
""",
    sha256 = "562a0e87eb413a7dcad29ebc8d578f6f97503473943585b051c1398a58189b06",
    strip_prefix = "TestParameterInjector-1.0",
    urls = [
        "https://mirror.bazel.build/github.com/google/TestParameterInjector/archive/v1.0.tar.gz",
        "https://github.com/google/TestParameterInjector/archive/v1.0.tar.gz",
    ],
)

# Used in src/main/java/com/google/devtools/build/lib/bazel/rules/java/jdk.WORKSPACE.
dist_http_archive(
    name = "rules_cc",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_FILE_WIN,
)

dist_http_archive(
    name = "rules_java",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_FILE_WIN,
)

dist_http_archive(
    name = "rules_proto",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE_WIN,
)

# For testing, have an distdir_tar with all the archives implicit in every
# WORKSPACE, to that they don't have to be refetched for every test
# calling `bazel sync`.
distdir_tar(
    name = "test_WORKSPACE_files",
    archives = [
        "android_tools_pkg-0.27.0.tar.gz",
        "r8-3.3.28.jar",
    ],
    dirname = "test_WORKSPACE/distdir",
    dist_deps = {dep: attrs for dep, attrs in DIST_DEPS.items() if "test_WORKSPACE_files" in attrs["used_in"]},
    sha256 = {
        "android_tools_pkg-0.27.0.tar.gz": "1afa4b7e13c82523c8b69e87f8d598c891ec7e2baa41d9e24e08becd723edb4d",
        "r8-3.3.28.jar": "8626ca32fb47aba7fddd2c897615e2e8ffcdb4d4b213572a2aefb3f838f01972",
    },
    urls = {
        "android_tools_pkg-0.27.0.tar.gz": [
            "https://mirror.bazel.build/bazel_android_tools/android_tools_pkg-0.27.0.tar.gz",
        ],
        "r8-3.3.28.jar": [
            "https://maven.google.com/com/android/tools/r8/3.3.28/r8-3.3.28.jar",
        ],
    },
)

dist_http_archive(
    name = "io_bazel_skydoc",
)

dist_http_archive(
    name = "platforms",
)

# This must be kept in sync with src/main/java/com/google/devtools/build/lib/bazel/rules/android/android_remote_tools.WORKSPACE
# and tools/android/android_extensions.bzl
http_archive(
    name = "android_tools_for_testing",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_FILE_WIN,
    sha256 = "1afa4b7e13c82523c8b69e87f8d598c891ec7e2baa41d9e24e08becd723edb4d",  # DO_NOT_REMOVE_THIS_ANDROID_TOOLS_UPDATE_MARKER
    url = "https://mirror.bazel.build/bazel_android_tools/android_tools_pkg-0.27.0.tar.gz",
)

# This must be kept in sync with src/main/java/com/google/devtools/build/lib/bazel/rules/android/android_remote_tools.WORKSPACE
# and tools/android/android_extensions.bzl
http_jar(
    name = "android_gmaven_r8_for_testing",
    sha256 = "8626ca32fb47aba7fddd2c897615e2e8ffcdb4d4b213572a2aefb3f838f01972",
    url = "https://maven.google.com/com/android/tools/r8/3.3.28/r8-3.3.28.jar",
)

dist_http_archive(
    name = "remote_coverage_tools",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_FILE_WIN,
)

dist_http_archive(
    name = "remotejdk11_linux_for_testing",
    build_file = "@local_jdk//:BUILD.bazel",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE_WIN,
)

dist_http_archive(
    name = "remotejdk11_linux_aarch64_for_testing",
    build_file = "@local_jdk//:BUILD.bazel",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE_WIN,
)

dist_http_archive(
    name = "remotejdk11_linux_ppc64le_for_testing",
    build_file = "@local_jdk//:BUILD.bazel",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE_WIN,
)

dist_http_archive(
    name = "remotejdk11_linux_s390x_for_testing",
    build_file = "@local_jdk//:BUILD.bazel",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE_WIN,
)

dist_http_archive(
    name = "remotejdk11_macos_for_testing",
    build_file = "@local_jdk//:BUILD.bazel",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE_WIN,
)

dist_http_archive(
    name = "remotejdk11_macos_aarch64_for_testing",
    build_file = "@local_jdk//:BUILD.bazel",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE_WIN,
)

dist_http_archive(
    name = "remotejdk11_win_for_testing",
    build_file = "@local_jdk//:BUILD.bazel",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE_WIN,
)

dist_http_archive(
    name = "remotejdk11_win_arm64_for_testing",
    build_file = "@local_jdk//:BUILD.bazel",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE_WIN,
)

[
    dist_http_archive(
        name = "remotejdk%s_%s_for_testing" % (version, os),
        build_file = "@local_jdk//:BUILD.bazel",
        patch_cmds = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE,
        patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE_WIN,
    )
    for version in ("17", "18")
    for os in ("linux", "macos", "macos_aarch64", "win", "win_arm64")
]

# Used in src/main/java/com/google/devtools/build/lib/bazel/rules/java/jdk.WORKSPACE.
dist_http_archive(
    name = "remote_java_tools_for_testing",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_FILE_WIN,
)

# Used in src/main/java/com/google/devtools/build/lib/bazel/rules/java/jdk.WORKSPACE.
dist_http_archive(
    name = "remote_java_tools_linux_for_testing",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_FILE_WIN,
)

# Used in src/main/java/com/google/devtools/build/lib/bazel/rules/java/jdk.WORKSPACE.
dist_http_archive(
    name = "remote_java_tools_windows_for_testing",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_FILE_WIN,
)

# Used in src/main/java/com/google/devtools/build/lib/bazel/rules/java/jdk.WORKSPACE.
dist_http_archive(
    name = "remote_java_tools_darwin_for_testing",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_FILE_WIN,
)

# Used in src/test/shell/bazel/testdata/jdk_http_archives.
dist_http_archive(
    name = "remote_java_tools_test",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_FILE_WIN,
)

# Used in src/test/shell/bazel/testdata/jdk_http_archives.
dist_http_archive(
    name = "remote_java_tools_test_linux",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_FILE_WIN,
)

# Used in src/test/shell/bazel/testdata/jdk_http_archives.
dist_http_archive(
    name = "remote_java_tools_test_windows",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_FILE_WIN,
)

# Used in src/test/shell/bazel/testdata/jdk_http_archives.
dist_http_archive(
    name = "remote_java_tools_test_darwin",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_FILE_WIN,
)

dist_http_archive(
    name = "openjdk11_linux_archive",
    build_file_content = """
java_runtime(name = 'runtime', srcs =  glob(['**']), visibility = ['//visibility:public'])
exports_files(["WORKSPACE"], visibility = ["//visibility:public"])
""",
)

# This must be kept in sync with src/test/shell/bazel/testdata/jdk_http_archives.
dist_http_archive(
    name = "openjdk11_darwin_archive",
    build_file_content = """
java_runtime(name = 'runtime', srcs =  glob(['**']), visibility = ['//visibility:public'])
exports_files(["WORKSPACE"], visibility = ["//visibility:public"])
""",
)

# This must be kept in sync with src/test/shell/bazel/testdata/jdk_http_archives.
dist_http_archive(
    name = "openjdk11_darwin_aarch64_archive",
    build_file_content = """
java_runtime(name = 'runtime', srcs =  glob(['**']), visibility = ['//visibility:public'])
exports_files(["WORKSPACE"], visibility = ["//visibility:public"])
""",
)

# This must be kept in sync with src/test/shell/bazel/testdata/jdk_http_archives.
dist_http_archive(
    name = "openjdk11_windows_archive",
    build_file_content = """
java_runtime(name = 'runtime', srcs =  glob(['**']), visibility = ['//visibility:public'])
exports_files(["WORKSPACE"], visibility = ["//visibility:public"])
""",
)

# This must be kept in sync with src/test/shell/bazel/testdata/jdk_http_archives.
dist_http_archive(
    name = "openjdk11_windows_arm64_archive",
    build_file_content = """
java_runtime(name = 'runtime', srcs =  glob(['**']), visibility = ['//visibility:public'])
exports_files(["WORKSPACE"], visibility = ["//visibility:public"])
""",
)

# This must be kept in sync with src/test/shell/bazel/testdata/jdk_http_archives.
[
    dist_http_archive(
        name = "openjdk%s_%s_archive" % (version, os),
        build_file_content = """
java_runtime(name = 'runtime', srcs =  glob(['**']), visibility = ['//visibility:public'])
exports_files(["WORKSPACE"], visibility = ["//visibility:public"])
""",
    )
    for version in ("17", "18")
    for os in ("linux", "darwin", "darwin_aarch64", "windows", "windows_arm64")
]

load("@io_bazel_skydoc//:setup.bzl", "stardoc_repositories")

stardoc_repositories()

register_execution_platforms("//:default_host_platform")  # buildozer: disable=positional-args

# Tools for building deb, rpm and tar files.
dist_http_archive(
    name = "rules_pkg",
)

load("@rules_pkg//:deps.bzl", "rules_pkg_dependencies")

rules_pkg_dependencies()

# Toolchains for Resource Compilation (.rc files on Windows).
load("//src/main/res:winsdk_configure.bzl", "winsdk_configure")

winsdk_configure(name = "local_config_winsdk")

load("@local_config_winsdk//:toolchains.bzl", "register_local_rc_exe_toolchains")

register_local_rc_exe_toolchains()

register_toolchains("//src/main/res:empty_rc_toolchain")

dist_http_archive(
    name = "com_github_grpc_grpc",
)

# Override the abseil-cpp version defined in grpc_deps(), which doesn't work on latest macOS
# Fixes https://github.com/bazelbuild/bazel/issues/15168
dist_http_archive(
    name = "com_google_absl",
)

# for patching the "com_github_cncf_udpa" deps loaded by grpc_deps
dist_http_archive(
    name = "com_envoyproxy_protoc_gen_validate",
)

dist_http_archive(
    name = "com_github_cncf_udpa",
)

dist_http_archive(
    name = "com_google_googleapis",
)

dist_http_archive(
    name = "upb",
)

dist_http_archive(
    name = "bazel_gazelle",
)

dist_http_archive(
    name = "rules_jvm_external",
)

# Projects using gRPC as an external dependency must call both grpc_deps() and
# grpc_extra_deps().
load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")

grpc_deps()

load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")

grpc_extra_deps()

load("//tools/distributions/debian:deps.bzl", "debian_deps")

debian_deps()

load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")

bazel_skylib_workspace()

load("@rules_jvm_external//:repositories.bzl", "rules_jvm_external_deps")

rules_jvm_external_deps()

load("@rules_jvm_external//:setup.bzl", "rules_jvm_external_setup")

rules_jvm_external_setup()

load("@rules_jvm_external//:defs.bzl", "maven_install")

maven_install(
    artifacts = [
       "com.google.guava:guava:31.1-jre",
       "com.google.guava:guava-testlib:31.1-jre",
       "com.google.guava:failureaccess:1.0.1",
       "com.google.errorprone:error_prone_annotations:2.16",
       "com.google.errorprone:error_prone_type_annotations:2.16",
       "com.google.code.findbugs:jsr305:3.0.2",
       "com.google.j2objc:j2objc-annotations:1.3",
       "com.github.stephenc.jcip:jcip-annotations:1.0-1",
       "org.checkerframework:checker-qual:3.12.0",
    ],
    repositories = [
        "https://dl.google.com/android/maven2",
        "https://repo1.maven.org/maven2",
    ],
)
