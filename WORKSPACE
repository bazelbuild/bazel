workspace(name = "io_bazel")

load("//tools/build_defs/repo:http.bzl", "http_archive")
load("//:distdir.bzl", "dist_http_archive", "dist_http_jar", "distdir_tar")
load("//:distdir_deps.bzl", "DIST_DEPS")
load("//:repositories.bzl", "embedded_jdk_repositories")
load("//tools/jdk:jdk_build_file.bzl", "JDK_BUILD_TEMPLATE")

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

EXPORT_WORKSPACE_BAZEL_IN_BUILD_FILE = [
    "test -f BUILD && chmod u+w BUILD || true",
    "echo >> BUILD",
    "echo 'exports_files([\"WORKSPACE.bazel\"], visibility = [\"//visibility:public\"])' >> BUILD",
]

EXPORT_WORKSPACE_IN_BUILD_FILE_WIN = [
    "Add-Content -Path BUILD -Value \"`nexports_files([`\"WORKSPACE`\"], visibility = [`\"//visibility:public`\"])`n\" -Force",
]

EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE_WIN = [
    "Add-Content -Path BUILD.bazel -Value \"`nexports_files([`\"WORKSPACE`\"], visibility = [`\"//visibility:public`\"])`n\" -Force",
]

EXPORT_WORKSPACE_BAZEL_IN_BUILD_FILE_WIN = [
    "Add-Content -Path BUILD -Value \"`nexports_files([`\"WORKSPACE.bazel`\"], visibility = [`\"//visibility:public`\"])`n\" -Force",
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

# In order to run //src/test/shell/bazel:maven_starlark_test, follow the
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
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE_WIN,
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
        "android_tools_pkg-0.28.0.tar",
        # for android_gmaven_r8
        "r8-8.0.40.jar",
    ],
    dirname = "derived/distdir",
    dist_deps = {dep: attrs for dep, attrs in DIST_DEPS.items() if "additional_distfiles" in attrs["used_in"]},
    sha256 = {
        "android_tools_pkg-0.28.0.tar": "db3b02421ae974e0b33573f3e4f658d5f89cc9a0b42baae0ba2ac08e25c0720a",
        "r8-8.0.40.jar": "ab1379835c7d3e5f21f80347c3c81e2f762e0b9b02748ae5232c3afa14adf702",
    },
    urls = {
        "android_tools_pkg-0.28.0.tar": [
            "https://mirror.bazel.build/bazel_android_tools/android_tools_pkg-0.28.0.tar",
        ],
        "r8-8.0.40.jar": [
            "https://maven.google.com/com/android/tools/r8/8.0.40/r8-8.0.40.jar",
        ],
    },
)

embedded_jdk_repositories()

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
    name = "rules_python",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE_WIN,
)

load("@rules_python//python:repositories.bzl", "py_repositories")

py_repositories()

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
        "android_tools_pkg-0.28.0.tar",
    ],
    dirname = "test_WORKSPACE/distdir",
    dist_deps = {dep: attrs for dep, attrs in DIST_DEPS.items() if "test_WORKSPACE_files" in attrs["used_in"]},
    sha256 = {
        "android_tools_pkg-0.28.0.tar": "db3b02421ae974e0b33573f3e4f658d5f89cc9a0b42baae0ba2ac08e25c0720a",
    },
    urls = {
        "android_tools_pkg-0.28.0.tar": [
            "https://mirror.bazel.build/bazel_android_tools/android_tools_pkg-0.28.0.tar",
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
    sha256 = "db3b02421ae974e0b33573f3e4f658d5f89cc9a0b42baae0ba2ac08e25c0720a",  # DO_NOT_REMOVE_THIS_ANDROID_TOOLS_UPDATE_MARKER
    url = "https://mirror.bazel.build/bazel_android_tools/android_tools_pkg-0.28.0.tar",
)

# This is here to override the android_gmaven_r8 rule from
# src/main/java/com/google/devtools/build/lib/bazel/rules/android/android_remote_tools.WORKSPACE
# so that tests like src/test/java/com/google/devtools/build/android/r8:AllTests
# use the most recent version of R8 rather than the one might be referenced in a released
# version of bazel that might have an outdated android_remote_tools.WORKSPACE relative to the tests.
dist_http_jar(
    name = "android_gmaven_r8",
)

dist_http_archive(
    name = "remote_coverage_tools",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_FILE_WIN,
)

dist_http_archive(
    name = "remotejdk11_linux_for_testing",
    build_file_content = JDK_BUILD_TEMPLATE.format(RUNTIME_VERSION = 11),
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE_WIN,
)

dist_http_archive(
    name = "remotejdk11_linux_aarch64_for_testing",
    build_file_content = JDK_BUILD_TEMPLATE.format(RUNTIME_VERSION = 11),
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE_WIN,
)

dist_http_archive(
    name = "remotejdk11_linux_ppc64le_for_testing",
    build_file_content = JDK_BUILD_TEMPLATE.format(RUNTIME_VERSION = 11),
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE_WIN,
)

dist_http_archive(
    name = "remotejdk11_linux_s390x_for_testing",
    build_file_content = JDK_BUILD_TEMPLATE.format(RUNTIME_VERSION = 11),
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE_WIN,
)

dist_http_archive(
    name = "remotejdk11_macos_for_testing",
    build_file_content = JDK_BUILD_TEMPLATE.format(RUNTIME_VERSION = 11),
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE_WIN,
)

dist_http_archive(
    name = "remotejdk11_macos_aarch64_for_testing",
    build_file_content = JDK_BUILD_TEMPLATE.format(RUNTIME_VERSION = 11),
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE_WIN,
)

dist_http_archive(
    name = "remotejdk11_win_for_testing",
    build_file_content = JDK_BUILD_TEMPLATE.format(RUNTIME_VERSION = 11),
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE_WIN,
)

dist_http_archive(
    name = "remotejdk11_win_arm64_for_testing",
    build_file_content = JDK_BUILD_TEMPLATE.format(RUNTIME_VERSION = 11),
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE_WIN,
)

[
    dist_http_archive(
        name = "remotejdk%s_%s_for_testing" % (version, os),
        build_file_content = JDK_BUILD_TEMPLATE.format(RUNTIME_VERSION = version),
        patch_cmds = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE,
        patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE_WIN,
    )
    for version in ("17", "20")
    for os in ("linux", "macos", "macos_aarch64", "win") + (("linux_s390x", "win_arm64") if version != "20" else ())
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
    name = "remote_java_tools_darwin_x86_64_for_testing",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_FILE_WIN,
)

# Used in src/main/java/com/google/devtools/build/lib/bazel/rules/java/jdk.WORKSPACE.
dist_http_archive(
    name = "remote_java_tools_darwin_arm64_for_testing",
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
    name = "remote_java_tools_test_darwin_x86_64",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_FILE_WIN,
)

# Used in src/test/shell/bazel/testdata/jdk_http_archives.
dist_http_archive(
    name = "remote_java_tools_test_darwin_arm64",
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
    name = "openjdk11_linux_s390x_archive",
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
    for version in ("17", "20")
    for os in ("linux", "darwin", "darwin_aarch64", "windows") + (("linux_s390x", "windows_arm64") if version != "20" else ())
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

dist_http_archive(
    name = "rules_testing",
    patch_cmds = EXPORT_WORKSPACE_BAZEL_IN_BUILD_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_BAZEL_IN_BUILD_FILE_WIN,
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
load("@rules_jvm_external//:specs.bzl", "maven")

maven_install(
    artifacts = [
        "com.beust:jcommander:1.82",
        "com.github.ben-manes.caffeine:caffeine:3.0.5",
        "com.github.kevinstern:software-and-algorithms:1.0",
        "com.github.stephenc.jcip:jcip-annotations:1.0-1",
        "com.google.api-client:google-api-client-gson:1.35.2",
        "com.google.api-client:google-api-client:1.35.2",
        "com.google.auth:google-auth-library-credentials:1.6.0",
        "com.google.auth:google-auth-library-oauth2-http:1.6.0",
        "com.google.auto.service:auto-service-annotations:1.0.1",
        "com.google.auto.service:auto-service:1.0",
        "com.google.auto.value:auto-value-annotations:1.9",
        "com.google.auto.value:auto-value:1.8.2",
        "com.google.auto:auto-common:1.2.1",
        "com.google.code.findbugs:jsr305:3.0.2",
        "com.google.code.gson:gson:2.9.0",
        "com.google.code.java-allocation-instrumenter:java-allocation-instrumenter:3.3.0",
        "com.google.errorprone:error_prone_annotation:2.19.0",
        "com.google.errorprone:error_prone_annotations:2.19.0",
        "com.google.errorprone:error_prone_check_api:2.19.0",
        "com.google.errorprone:error_prone_core:2.19.0",
        "com.google.errorprone:error_prone_type_annotations:2.19.0",
        "com.google.flogger:flogger-system-backend:0.5.1",
        "com.google.flogger:flogger:0.5.1",
        "com.google.flogger:google-extensions:0.5.1",
        "com.google.guava:failureaccess:1.0.1",
        "com.google.guava:guava:31.1-jre",
        "com.google.http-client:google-http-client-gson:1.42.0",
        "com.google.http-client:google-http-client:1.42.0",
        "com.google.j2objc:j2objc-annotations:1.3",
        "com.ryanharter.auto.value:auto-value-gson-extension:1.3.1",
        "com.ryanharter.auto.value:auto-value-gson-runtime:1.3.1",
        "com.ryanharter.auto.value:auto-value-gson-factory:1.3.1",
        "com.squareup:javapoet:1.12.0",
        "commons-collections:commons-collections:3.2.2",
        "commons-lang:commons-lang:2.6",
        "io.github.java-diff-utils:java-diff-utils:4.0",
        "io.grpc:grpc-api:1.48.1",
        "io.grpc:grpc-auth:1.48.1",
        "io.grpc:grpc-context:1.48.1",
        "io.grpc:grpc-core:1.48.1",
        "io.grpc:grpc-netty:1.48.1",
        "io.grpc:grpc-protobuf-lite:1.48.1",
        "io.grpc:grpc-protobuf:1.48.1",
        "io.grpc:grpc-stub:1.48.1",
        "io.netty:netty-buffer:4.1.93.Final",
        "io.netty:netty-codec-http2:4.1.93.Final",
        "io.netty:netty-codec-http:4.1.93.Final",
        "io.netty:netty-codec:4.1.93.Final",
        "io.netty:netty-common:4.1.93.Final",
        "io.netty:netty-handler-proxy:4.1.93.Final",
        "io.netty:netty-handler:4.1.93.Final",
        "io.netty:netty-resolver-dns:4.1.93.Final",
        "io.netty:netty-resolver:4.1.93.Final",
        "io.netty:netty-tcnative-boringssl-static:jar:linux-aarch_64:2.0.56.Final",
        "io.netty:netty-tcnative-boringssl-static:jar:linux-x86_64:2.0.56.Final",
        "io.netty:netty-tcnative-boringssl-static:jar:osx-aarch_64:2.0.56.Final",
        "io.netty:netty-tcnative-boringssl-static:jar:osx-x86_64:2.0.56.Final",
        "io.netty:netty-tcnative-boringssl-static:jar:windows-x86_64:2.0.56.Final",
        "io.netty:netty-tcnative-classes:2.0.56.Final",
        "io.netty:netty-transport-classes-epoll:4.1.93.Final",
        "io.netty:netty-transport-classes-kqueue:4.1.93.Final",
        "io.netty:netty-transport-native-epoll:jar:linux-aarch_64:4.1.93.Final",
        "io.netty:netty-transport-native-epoll:jar:linux-x86_64:4.1.93.Final",
        "io.netty:netty-transport-native-kqueue:jar:osx-aarch_64:4.1.93.Final",
        "io.netty:netty-transport-native-kqueue:jar:osx-x86_64:4.1.93.Final",
        "io.netty:netty-transport-native-unix-common:4.1.93.Final",
        "io.netty:netty-transport-native-unix-common:jar:linux-aarch_64:4.1.93.Final",
        "io.netty:netty-transport-native-unix-common:jar:linux-x86_64:4.1.93.Final",
        "io.netty:netty-transport-native-unix-common:jar:osx-aarch_64:4.1.93.Final",
        "io.netty:netty-transport-native-unix-common:jar:osx-x86_64:4.1.93.Final",
        "io.netty:netty-transport-sctp:4.1.93.Final",
        "io.netty:netty-transport:4.1.93.Final",
        "io.reactivex.rxjava3:rxjava:3.1.2",
        "javax.activation:javax.activation-api:1.2.0",
        "javax.annotation:javax.annotation-api:1.3.2",
        "javax.inject:javax.inject:1",
        "net.bytebuddy:byte-buddy-agent:1.11.13",
        "net.bytebuddy:byte-buddy:1.11.13",
        "org.apache.commons:commons-compress:1.19",
        "org.apache.commons:commons-pool2:2.8.0",
        "org.apache.tomcat:tomcat-annotations-api:8.0.5",
        "org.apache.velocity:velocity:1.7",
        "org.checkerframework:checker-qual:3.19.0",
        "org.ow2.asm:asm-analysis:9.2",
        "org.ow2.asm:asm-commons:9.2",
        "org.ow2.asm:asm-tree:9.2",
        "org.ow2.asm:asm-util:9.2",
        "org.ow2.asm:asm:9.2",
        "org.pcollections:pcollections:3.1.4",
        "org.threeten:threeten-extra:1.5.0",
        "org.tukaani:xz:1.9",
        "tools.profiler:async-profiler:2.9",
        # The following jars are for testing.
        # junit is not test only due to //src/java_tools/junitrunner/java/com/google/testing/junit/junit4:runner,
        # and hamcrest is a dependency of junit.
        "junit:junit:4.13.2",
        "org.hamcrest:hamcrest-core:1.3",
        maven.artifact(
            "com.google.guava",
            "guava-testlib",
            "31.1-jre",
            testonly = True,
        ),
        maven.artifact(
            "com.google.jimfs",
            "jimfs",
            "1.2",
            testonly = True,
        ),
        maven.artifact(
            "com.google.testing.compile",
            "compile-testing",
            "0.18",
            testonly = True,
        ),
        maven.artifact(
            "com.google.truth",
            "truth",
            "1.1.3",
            testonly = True,
        ),
        maven.artifact(
            "com.google.truth.extensions",
            "truth-java8-extension",
            "1.1.3",
            testonly = True,
        ),
        maven.artifact(
            "com.google.truth.extensions",
            "truth-liteproto-extension",
            "1.1.3",
            testonly = True,
        ),
        maven.artifact(
            "com.google.truth.extensions",
            "truth-proto-extension",
            "1.1.3",
            testonly = True,
        ),
        maven.artifact(
            "org.mockito",
            "mockito-core",
            "3.12.4",
            testonly = True,
        ),
    ],
    excluded_artifacts = [
        # org.apache.httpcomponents and org.eclipse.jgit:org.eclipse.jgit
        # require java.security.jgss module to be embedded in the Bazel binary.
        "org.apache.httpcomponents:httpclient",
        "org.apache.httpcomponents:httpcore",
        "org.eclipse.jgit:org.eclipse.jgit",
        # We build protobuf Java library from source, exclude protobuf jars to be safe.
        "com.google.protobuf:protobuf-java",
        "com.google.protobuf:protobuf-javalite",
    ],
    fail_if_repin_required = False,
    maven_install_json = "//:maven_install.json",
    repositories = [
        "https://repo1.maven.org/maven2",
    ],
    strict_visibility = True,
)

load("@maven//:defs.bzl", "pinned_maven_install")

pinned_maven_install()

maven_install(
    name = "maven_android",
    artifacts = [
        "androidx.databinding:databinding-compiler:3.4.0-alpha10",
        "com.android.tools.build:builder:7.1.3",
        "com.android.tools.build:manifest-merger:30.1.3",
        "com.android.tools:sdk-common:30.1.3",
        "com.android.tools:annotations:30.1.3",
        "com.android.tools.layoutlib:layoutlib-api:30.1.3",
        "com.android.tools:common:30.1.3",
        "com.android.tools:repository:30.1.3",
    ],
    fail_if_repin_required = True,
    maven_install_json = "//src/tools/android:maven_android_install.json",
    repositories = [
        "https://dl.google.com/android/maven2",
        "https://repo1.maven.org/maven2",
    ],
)

load("@maven_android//:defs.bzl", pinned_maven_install_android = "pinned_maven_install")

pinned_maven_install_android()
