workspace(name = "io_bazel")

load("//tools/build_defs/repo:http.bzl", "http_archive", "http_file")
load("//:distdir.bzl", "dist_http_archive", "distdir_tar")
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
    actual = "//third_party/grpc:grpc-java-plugin",
)

bind(
    name = "grpc-jar",
    actual = "//third_party/grpc:grpc-jar",
)

bind(
    name = "guava",
    actual = "//third_party:guava",
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
        "android_tools_pkg-0.23.0.tar.gz",
    ],
    dirname = "derived/distdir",
    dist_deps = {dep: attrs for dep, attrs in DIST_DEPS.items() if "additional_distfiles" in attrs["used_in"]},
    sha256 = {
        "android_tools_pkg-0.23.0.tar.gz": "ed5290594244c2eeab41f0104519bcef51e27c699ff4b379fcbd25215270513e",
    },
    urls = {
        "android_tools_pkg-0.23.0.tar.gz": [
            "https://mirror.bazel.build/bazel_android_tools/android_tools_pkg-0.23.0.tar.gz",
        ],
    },
)

# OpenJDK distributions used to create a version of Bazel bundled with the OpenJDK.
http_file(
    name = "openjdk_linux",
    downloaded_file_path = "zulu-linux.tar.gz",
    sha256 = "65bfe4e0ffa74a680ee4410db46b17e30cd9397b664a92a886599fe1f3530969",
    urls = ["https://mirror.bazel.build/openjdk/azul-zulu11.37.17-ca-jdk11.0.6/zulu11.37.17-ca-jdk11.0.6-linux_x64-linux_x64-allmodules-b23d4e05466f2aa1fdcd72d3d3a8e962206b64bf-1581689070.tar.gz"],
)

http_file(
    name = "openjdk_linux_vanilla",
    downloaded_file_path = "zulu-linux-vanilla.tar.gz",
    sha256 = "360626cc19063bc411bfed2914301b908a8f77a7919aaea007a977fa8fb3cde1",
    urls = ["https://mirror.bazel.build/openjdk/azul-zulu11.37.17-ca-jdk11.0.6/zulu11.37.17-ca-jdk11.0.6-linux_x64.tar.gz"],
)

http_file(
    name = "openjdk_linux_minimal",
    downloaded_file_path = "zulu-linux-minimal.tar.gz",
    sha256 = "91f7d52f695c681d4e21499b4319d548aadef249a6b3053e306308992e1e29ae",
    urls = ["https://mirror.bazel.build/openjdk/azul-zulu11.37.17-ca-jdk11.0.6/zulu11.37.17-ca-jdk11.0.6-linux_x64-minimal-b23d4e05466f2aa1fdcd72d3d3a8e962206b64bf-1581689068.tar.gz"],
)

http_file(
    name = "openjdk_linux_aarch64",
    downloaded_file_path = "zulu-linux-aarch64.tar.gz",
    sha256 = "6b245793087300db3ee82ab0d165614f193a73a60f2f011e347756c1e6ca5bac",
    urls = ["https://mirror.bazel.build/openjdk/azul-zulu11.37.48-ca-jdk11.0.6/zulu11.37.48-ca-jdk11.0.6-linux_aarch64-allmodules-b23d4e05466f2aa1fdcd72d3d3a8e962206b64bf-1581690750.tar.gz"],
)

http_file(
    name = "openjdk_linux_aarch64_vanilla",
    downloaded_file_path = "zulu-linux-aarch64-vanilla.tar.gz",
    sha256 = "a452f1b9682d9f83c1c14e54d1446e1c51b5173a3a05dcb013d380f9508562e4",
    urls = ["https://mirror.bazel.build/openjdk/azul-zulu11.37.48-ca-jdk11.0.6/zulu11.37.48-ca-jdk11.0.6-linux_aarch64.tar.gz"],
)

http_file(
    name = "openjdk_linux_aarch64_minimal",
    downloaded_file_path = "zulu-linux-aarch64-minimal.tar.gz",
    sha256 = "06f6520a877704c77614bcfc4f846cc7cbcbf5eaad149bf7f19f4f16e285c9de",
    urls = ["https://mirror.bazel.build/openjdk/azul-zulu11.37.48-ca-jdk11.0.6/zulu11.37.48-ca-jdk11.0.6-linux_aarch64-minimal-b23d4e05466f2aa1fdcd72d3d3a8e962206b64bf-1581690750.tar.gz"],
)

http_file(
    name = "openjdk_linux_ppc64le_vanilla",
    downloaded_file_path = "adoptopenjdk-ppc64le-vanilla.tar.gz",
    sha256 = "a417db0295b1f4b538ecbaf7c774f3a177fab9657a665940170936c0eca4e71a",
    urls = [
        "https://mirror.bazel.build/openjdk/AdoptOpenJDK/openjdk11-binaries/releases/download/jdk-11.0.7+10/OpenJDK11U-jdk_ppc64le_linux_hotspot_11.0.7_10.tar.gz",
        "https://github.com/AdoptOpenJDK/openjdk11-binaries/releases/download/jdk-11.0.7+10/OpenJDK11U-jdk_ppc64le_linux_hotspot_11.0.7_10.tar.gz",
    ],
)

http_file(
    name = "openjdk_linux_s390x_vanilla",
    downloaded_file_path = "adoptopenjdk-s390x-vanilla.tar.gz",
    sha256 = "d9b72e87a1d3ebc0c9552f72ae5eb150fffc0298a7cb841f1ce7bfc70dcd1059",
    urls = [
        "https://mirror.bazel.build/github.com/AdoptOpenJDK/openjdk11-binaries/releases/download/jdk-11.0.7+10/OpenJDK11U-jdk_s390x_linux_hotspot_11.0.7_10.tar.gz",
        "https://github.com/AdoptOpenJDK/openjdk11-binaries/releases/download/jdk-11.0.7+10/OpenJDK11U-jdk_s390x_linux_hotspot_11.0.7_10.tar.gz",
    ],
)

http_file(
    name = "openjdk_macos_x86_64",
    downloaded_file_path = "zulu-macos.tar.gz",
    sha256 = "8e283cfd23c7555be8e17295ed76eb8f00324c88ab904b8de37bbe08f90e569b",
    urls = ["https://mirror.bazel.build/openjdk/azul-zulu11.37.17-ca-jdk11.0.6/zulu11.37.17-ca-jdk11.0.6-macosx_x64-allmodules-b23d4e05466f2aa1fdcd72d3d3a8e962206b64bf-1581689066.tar.gz"],
)

http_file(
    name = "openjdk_macos_x86_64_vanilla",
    downloaded_file_path = "zulu-macos-vanilla.tar.gz",
    sha256 = "e1fe56769f32e2aaac95e0a8f86b5a323da5af3a3b4bba73f3086391a6cc056f",
    urls = ["https://mirror.bazel.build/openjdk/azul-zulu11.37.17-ca-jdk11.0.6/zulu11.37.17-ca-jdk11.0.6-macosx_x64.tar.gz"],
)

http_file(
    name = "openjdk_macos_x86_64_minimal",
    downloaded_file_path = "zulu-macos-minimal.tar.gz",
    sha256 = "1bacb1c07035d4066d79f0b65b4ea0ebd1954f3662bdfe3618da382ac8fd23a6",
    urls = ["https://mirror.bazel.build/openjdk/azul-zulu11.37.17-ca-jdk11.0.6/zulu11.37.17-ca-jdk11.0.6-macosx_x64-minimal-b23d4e05466f2aa1fdcd72d3d3a8e962206b64bf-1581689063.tar.gz"],
)

http_file(
    name = "openjdk_macos_aarch64",
    downloaded_file_path = "zulu-macos-aarch64.tar.gz",
    sha256 = "a900ef793cb34b03ac5d93ea2f67291b6842e99d500934e19393a8d8f9bfa6ff",
    urls = ["https://mirror.bazel.build/openjdk/azul-zulu11.45.27-ca-jdk11.0.10/zulu11.45.27-ca-jdk11.0.10-macosx_aarch64-allmodules-1611665569.tar.gz"],
)

http_file(
    name = "openjdk_macos_aarch64_vanilla",
    downloaded_file_path = "zulu-macos-aarch64-vanilla.tar.gz",
    sha256 = "3dcc636e64ae58b922269c2dc9f20f6f967bee90e3f6847d643c4a566f1e8d8a",
    urls = [
        "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu11.45.27-ca-jdk11.0.10-macosx_aarch64.tar.gz",
        "https://cdn.azul.com/zulu/bin/zulu11.45.27-ca-jdk11.0.10-macosx_aarch64.tar.gz",
    ],
)

http_file(
    name = "openjdk_macos_aarch64_minimal",
    downloaded_file_path = "zulu-macos-aarch64-minimal.tar.gz",
    sha256 = "f4f606926e6deeaa8b8397e299313d9df87642fe464b0ccf1ed0432aeb00640b",
    urls = ["https://mirror.bazel.build/openjdk/azul-zulu11.45.27-ca-jdk11.0.10/zulu11.45.27-ca-jdk11.0.10-macosx_aarch64-minimal-1611665562.tar.gz"],
)

http_file(
    name = "openjdk_win",
    downloaded_file_path = "zulu-win.zip",
    sha256 = "8e1604b3a27dcf639bc6d1a73103f1211848139e4cceb081d0a74a99e1e6f995",
    urls = ["https://mirror.bazel.build/openjdk/azul-zulu11.37.17-ca-jdk11.0.6/zulu11.37.17-ca-jdk11.0.6-win_x64-allmodules-b23d4e05466f2aa1fdcd72d3d3a8e962206b64bf-1581689080.zip"],
)

http_file(
    name = "openjdk_win_vanilla",
    downloaded_file_path = "zulu-win-vanilla.zip",
    sha256 = "a9695617b8374bfa171f166951214965b1d1d08f43218db9a2a780b71c665c18",
    urls = ["https://mirror.bazel.build/openjdk/azul-zulu11.37.17-ca-jdk11.0.6/zulu11.37.17-ca-jdk11.0.6-win_x64.zip"],
)

http_file(
    name = "openjdk_win_minimal",
    downloaded_file_path = "zulu-win-minimal.zip",
    sha256 = "b90a713c9c2d9ea23cad44d2c2dfcc9af22faba9bde55dedc1c3bb9f556ac1ae",
    urls = ["https://mirror.bazel.build/openjdk/azul-zulu11.37.17-ca-jdk11.0.6/zulu11.37.17-ca-jdk11.0.6-win_x64-minimal-b23d4e05466f2aa1fdcd72d3d3a8e962206b64bf-1581689080.zip"],
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
    sha256 = "9dc9157a9a1551ec7a7e43daea9a694a0bb5fb8bec81235d8a1e6ef64c716dcb",
    strip_prefix = "googletest-release-1.10.0",
    urls = [
        "https://mirror.bazel.build/github.com/google/googletest/archive/release-1.10.0.tar.gz",
        "https://github.com/google/googletest/archive/release-1.10.0.tar.gz",
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
    strip_prefix = "zstd-jni-1.5.0-4",
)

http_archive(
    name = "org_snakeyaml",
    build_file_content = """
java_library(
    name = "snakeyaml",
    testonly = True,
    srcs = glob(["src/main/**/*.java"]),
    visibility = ["@com_google_testparameterinjector//:__pkg__"],
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
        "zulu11.50.19-ca-jdk11.0.12-linux_x64.tar.gz",
        "zulu11.50.19-ca-jdk11.0.12-linux_aarch64.tar.gz",
        "zulu11.50.19-ca-jdk11.0.12-macosx_x64.tar.gz",
        "zulu11.50.19-ca-jdk11.0.12-macosx_aarch64.tar.gz",
        "zulu11.50.19-ca-jdk11.0.12-win_x64.tar.gz",
        "android_tools_pkg-0.23.0.tar.gz",
    ],
    dirname = "test_WORKSPACE/distdir",
    dist_deps = {dep: attrs for dep, attrs in DIST_DEPS.items() if "test_WORKSPACE_files" in attrs["used_in"]},
    sha256 = {
        "zulu11.50.19-ca-jdk11.0.12-linux_x64.tar.gz": "b8e8a63b79bc312aa90f3558edbea59e71495ef1a9c340e38900dd28a1c579f3",
        "zulu11.50.19-ca-jdk11.0.12-linux_aarch64.tar.gz": "61254688067454d3ccf0ef25993b5dcab7b56c8129e53b73566c28a8dd4d48fb",
        "zulu11.50.19-ca-jdk11.0.12-macosx_x64.tar.gz": "0b8c8b7cf89c7c55b7e2239b47201d704e8d2170884875b00f3103cf0662d6d7",
        "zulu11.50.19-ca-jdk11.0.12-macosx_aarch64.tar.gz": "e908a0b4c0da08d41c3e19230f819b364ff2e5f1dafd62d2cf991a85a34d3a17",
        "zulu11.50.19-ca-jdk11.0.12-win_x64.tar.gz": "42ae65e75d615a3f06a674978e1fa85fdf078cad94e553fee3e779b2b42bb015",
        "android_tools_pkg-0.23.0.tar.gz": "ed5290594244c2eeab41f0104519bcef51e27c699ff4b379fcbd25215270513e",
    },
    urls = {
        "zulu11.50.19-ca-jdk11.0.12-linux_x64.tar.gz": ["https://mirror.bazel.build/openjdk/azul-zulu11.50.19-ca-jdk11.0.12/zulu11.50.19-ca-jdk11.0.12-linux_x64.tar.gz"],
        "zulu11.50.19-ca-jdk11.0.12-linux_aarch64.tar.gz": ["https://mirror.bazel.build/openjdk/azul-zulu11.50.19-ca-jdk11.0.12/zulu11.50.19-ca-jdk11.0.12-linux_aarch64.tar.gz"],
        "zulu11.50.19-ca-jdk11.0.12-macosx_x64.tar.gz": ["https://mirror.bazel.build/openjdk/azul-zulu11.50.19-ca-jdk11.0.12/zulu11.50.19-ca-jdk11.0.12-macosx_x64.tar.gz"],
        "zulu11.50.19-ca-jdk11.0.12-macosx_aarch64.tar.gz": ["https://mirror.bazel.build/openjdk/azul-zulu11.50.19-ca-jdk11.0.12/zulu11.50.19-ca-jdk11.0.12-macosx_aarch64.tar.gz"],
        "zulu11.50.19-ca-jdk11.0.12-win_x64.tar.gz": ["https://mirror.bazel.build/openjdk/azul-zulu11.50.19-ca-jdk11.0.12/zulu11.50.19-ca-jdk11.0.12-win_x64.zip"],
        "android_tools_pkg-0.23.0.tar.gz": [
            "https://mirror.bazel.build/bazel_android_tools/android_tools_pkg-0.23.0.tar.gz",
        ],
    },
)

dist_http_archive(
    name = "io_bazel_skydoc",
)

load("//scripts/docs:doc_versions.bzl", "DOC_VERSIONS")

# Load versioned documentation tarballs from GCS
[http_file(
    # Split on "-" to get the version without cherrypick commits.
    name = "jekyll_tree_%s" % DOC_VERSION["version"].split("-")[0].replace(".", "_"),
    sha256 = DOC_VERSION["sha256"],
    urls = ["https://mirror.bazel.build/bazel_versioned_docs/jekyll-tree-%s.tar" % DOC_VERSION["version"]],
) for DOC_VERSION in DOC_VERSIONS]

# Load shared base CSS theme from bazelbuild/bazel-website
http_archive(
    name = "bazel_website",
    # TODO(https://github.com/bazelbuild/bazel/issues/10793)
    # - Export files from bazel-website's BUILD, instead of doing it here.
    # - Share more common stylesheets, like footer and navbar.
    build_file_content = """
exports_files(["_sass/style.scss"])
""",
    sha256 = "a5f531dd1d62e6947dcfc279656ffc2fdf6f447c163914c5eabf7961b4cb6eb4",
    strip_prefix = "bazel-website-c174fa288aa079b68416d2ce2cc97268fa172f42",
    urls = ["https://github.com/bazelbuild/bazel-website/archive/c174fa288aa079b68416d2ce2cc97268fa172f42.tar.gz"],
)

# Stardoc recommends declaring its dependencies via "*_dependencies" functions.
# This requires that the repositories these functions come from need to be
# fetched unconditionally for everything (including just building bazel!), so
# provide them as http_archives that can be shiped in the distdir, to keep the
# distribution archive self-contained.
dist_http_archive(
    name = "io_bazel_rules_sass",
)

dist_http_archive(
    name = "build_bazel_rules_nodejs",
)

http_archive(
    name = "java_tools_langtools_javac11",
    sha256 = "cf0814fa002ef3d794582bb086516d8c9ed0958f83f19799cdb08949019fe4c7",
    urls = [
        "https://mirror.bazel.build/bazel_java_tools/jdk_langtools/langtools_jdk11_v2.zip",
    ],
)

dist_http_archive(
    name = "platforms",
)

# This must be kept in sync with src/main/java/com/google/devtools/build/lib/bazel/rules/android/android_remote_tools.WORKSPACE
http_archive(
    name = "android_tools_for_testing",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_FILE_WIN,
    sha256 = "ed5290594244c2eeab41f0104519bcef51e27c699ff4b379fcbd25215270513e",  # DO_NOT_REMOVE_THIS_ANDROID_TOOLS_UPDATE_MARKER
    url = "https://mirror.bazel.build/bazel_android_tools/android_tools_pkg-0.23.0.tar.gz",
)

# Used in src/main/java/com/google/devtools/build/lib/bazel/rules/coverage.WORKSPACE.
dist_http_archive(
    name = "remote_coverage_tools",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_FILE_WIN,
)

# This must be kept in sync with src/main/java/com/google/devtools/build/lib/bazel/rules/java/jdk.WORKSPACE.
http_archive(
    name = "remotejdk11_linux_for_testing",
    build_file = "@local_jdk//:BUILD.bazel",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE_WIN,
    sha256 = "b8e8a63b79bc312aa90f3558edbea59e71495ef1a9c340e38900dd28a1c579f3",
    strip_prefix = "zulu11.50.19-ca-jdk11.0.12-linux_x64",
    urls = ["https://mirror.bazel.build/openjdk/azul-zulu11.50.19-ca-jdk11.0.12/zulu11.50.19-ca-jdk11.0.12-linux_x64.tar.gz"],
)

# This must be kept in sync with src/main/java/com/google/devtools/build/lib/bazel/rules/java/jdk.WORKSPACE.
http_archive(
    name = "remotejdk11_linux_aarch64_for_testing",
    build_file = "@local_jdk//:BUILD.bazel",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE_WIN,
    sha256 = "61254688067454d3ccf0ef25993b5dcab7b56c8129e53b73566c28a8dd4d48fb",
    strip_prefix = "zulu11.50.19-ca-jdk11.0.12-linux_aarch64",
    urls = ["https://mirror.bazel.build/openjdk/azul-zulu11.50.19-ca-jdk11.0.12/zulu11.50.19-ca-jdk11.0.12-linux_aarch64.tar.gz"],
)

# This must be kept in sync with src/main/java/com/google/devtools/build/lib/bazel/rules/java/jdk.WORKSPACE.
http_archive(
    name = "remotejdk11_linux_ppc64le_for_testing",
    build_file = "@local_jdk//:BUILD.bazel",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE_WIN,
    sha256 = "a417db0295b1f4b538ecbaf7c774f3a177fab9657a665940170936c0eca4e71a",
    strip_prefix = "jdk-11.0.7+10",
    urls = [
        "https://mirror.bazel.build/openjdk/AdoptOpenJDK/openjdk11-binaries/releases/download/jdk-11.0.7+10/OpenJDK11U-jdk_ppc64le_linux_hotspot_11.0.7_10.tar.gz",
        "https://github.com/AdoptOpenJDK/openjdk11-binaries/releases/download/jdk-11.0.7+10/OpenJDK11U-jdk_ppc64le_linux_hotspot_11.0.7_10.tar.gz",
    ],
)

# This must be kept in sync with src/main/java/com/google/devtools/build/lib/bazel/rules/java/jdk.WORKSPACE.
http_archive(
    name = "remotejdk11_linux_s390x_for_testing",
    build_file = "@local_jdk//:BUILD.bazel",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE_WIN,
    sha256 = "d9b72e87a1d3ebc0c9552f72ae5eb150fffc0298a7cb841f1ce7bfc70dcd1059",
    strip_prefix = "jdk-11.0.7+10",
    urls = [
        "https://mirror.bazel.build/github.com/AdoptOpenJDK/openjdk11-binaries/releases/download/jdk-11.0.7+10/OpenJDK11U-jdk_s390x_linux_hotspot_11.0.7_10.tar.gz",
        "https://github.com/AdoptOpenJDK/openjdk11-binaries/releases/download/jdk-11.0.7+10/OpenJDK11U-jdk_s390x_linux_hotspot_11.0.7_10.tar.gz",
    ],
)

# This must be kept in sync with src/main/java/com/google/devtools/build/lib/bazel/rules/java/jdk.WORKSPACE.
http_archive(
    name = "remotejdk11_macos_for_testing",
    build_file = "@local_jdk//:BUILD.bazel",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE_WIN,
    sha256 = "0b8c8b7cf89c7c55b7e2239b47201d704e8d2170884875b00f3103cf0662d6d7",
    strip_prefix = "zulu11.50.19-ca-jdk11.0.12-macosx_x64",
    urls = ["https://mirror.bazel.build/openjdk/azul-zulu11.50.19-ca-jdk11.0.12/zulu11.50.19-ca-jdk11.0.12-macosx_x64.tar.gz"],
)

# This must be kept in sync with src/main/java/com/google/devtools/build/lib/bazel/rules/java/jdk.WORKSPACE.
http_archive(
    name = "remotejdk11_macos_aarch64_for_testing",
    build_file = "@local_jdk//:BUILD.bazel",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE_WIN,
    sha256 = "e908a0b4c0da08d41c3e19230f819b364ff2e5f1dafd62d2cf991a85a34d3a17",
    strip_prefix = "zulu11.50.19-ca-jdk11.0.12-macosx_aarch64",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu11.50.19-ca-jdk11.0.12/zulu11.50.19-ca-jdk11.0.12-macosx_aarch64.tar.gz",
        "https://cdn.azul.com/zulu/bin/zulu11.50.19-ca-jdk11.0.12-macosx_aarch64.tar.gz",
    ],
)

# This must be kept in sync with src/main/java/com/google/devtools/build/lib/bazel/rules/java/jdk.WORKSPACE.
http_archive(
    name = "remotejdk11_win_for_testing",
    build_file = "@local_jdk//:BUILD.bazel",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE_WIN,
    sha256 = "42ae65e75d615a3f06a674978e1fa85fdf078cad94e553fee3e779b2b42bb015",
    strip_prefix = "zulu11.50.19-ca-jdk11.0.12-win_x64",
    urls = ["https://mirror.bazel.build/openjdk/azul-zulu11.50.19-ca-jdk11.0.12/zulu11.50.19-ca-jdk11.0.12-win_x64.zip"],
)

# This must be kept in sync with src/main/java/com/google/devtools/build/lib/bazel/rules/java/jdk.WORKSPACE.
http_archive(
    name = "remotejdk17_linux_for_testing",
    build_file = "@local_jdk//:BUILD.bazel",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE_WIN,
    sha256 = "37c4f8e48536cceae8c6c20250d6c385e176972532fd35759fa7d6015c965f56",
    strip_prefix = "zulu17.28.13-ca-jdk17.0.0-linux_x64",
    urls = [
        "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu17.28.13-ca-jdk17.0.0-linux_x64.tar.gz",
        "https://cdn.azul.com/zulu/bin/zulu17.28.13-ca-jdk17.0.0-linux_x64.tar.gz",
    ],
)

# This must be kept in sync with src/main/java/com/google/devtools/build/lib/bazel/rules/java/jdk.WORKSPACE.
http_archive(
    name = "remotejdk17_macos_for_testing",
    build_file = "@local_jdk//:BUILD.bazel",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE_WIN,
    sha256 = "6029b1fe6853cecad22ab99ac0b3bb4fb8c903dd2edefa91c3abc89755bbd47d",
    strip_prefix = "zulu17.28.13-ca-jdk17.0.0-macosx_x64",
    urls = [
        "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu17.28.13-ca-jdk17.0.0-macosx_x64.tar.gz",
        "https://cdn.azul.com/zulu/bin/zulu17.28.13-ca-jdk17.0.0-macosx_x64.tar.gz",
    ],
)

# This must be kept in sync with src/main/java/com/google/devtools/build/lib/bazel/rules/java/jdk.WORKSPACE.
http_archive(
    name = "remotejdk17_macos_aarch64_for_testing",
    build_file = "@local_jdk//:BUILD.bazel",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE_WIN,
    sha256 = "6b17f01f767ee7abf4704149ca4d86423aab9b16b68697b7d36e9b616846a8b0",
    strip_prefix = "zulu17.28.13-ca-jdk17.0.0-macosx_aarch64",
    urls = [
        "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu17.28.13-ca-jdk17.0.0-macosx_aarch64.tar.gz",
        "https://cdn.azul.com/zulu/bin/zulu17.28.13-ca-jdk17.0.0-macosx_aarch64.tar.gz",
    ],
)

# This must be kept in sync with src/main/java/com/google/devtools/build/lib/bazel/rules/java/jdk.WORKSPACE.
http_archive(
    name = "remotejdk17_win_for_testing",
    build_file = "@local_jdk//:BUILD.bazel",
    patch_cmds = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE,
    patch_cmds_win = EXPORT_WORKSPACE_IN_BUILD_BAZEL_FILE_WIN,
    sha256 = "f4437011239f3f0031c794bb91c02a6350bc941d4196bdd19c9f157b491815a3",
    strip_prefix = "zulu17.28.13-ca-jdk17.0.0-win_x64",
    urls = [
        "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu17.28.13-ca-jdk17.0.0-win_x64.zip",
        "https://cdn.azul.com/zulu/bin/zulu17.28.13-ca-jdk17.0.0-win_x64.zip",
    ],
)

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

# This must be kept in sync with src/test/shell/bazel/testdata/jdk_http_archives.
http_archive(
    name = "openjdk11_linux_archive",
    build_file_content = """
java_runtime(name = 'runtime', srcs =  glob(['**']), visibility = ['//visibility:public'])
exports_files(["WORKSPACE"], visibility = ["//visibility:public"])
""",
    sha256 = "b8e8a63b79bc312aa90f3558edbea59e71495ef1a9c340e38900dd28a1c579f3",
    strip_prefix = "zulu11.50.19-ca-jdk11.0.12-linux_x64",
    urls = ["https://mirror.bazel.build/openjdk/azul-zulu11.50.19-ca-jdk11.0.12/zulu11.50.19-ca-jdk11.0.12-linux_x64.tar.gz"],
)

# This must be kept in sync with src/test/shell/bazel/testdata/jdk_http_archives.
http_archive(
    name = "openjdk11_darwin_archive",
    build_file_content = """
java_runtime(name = 'runtime', srcs =  glob(['**']), visibility = ['//visibility:public'])
exports_files(["WORKSPACE"], visibility = ["//visibility:public"])
""",
    sha256 = "0b8c8b7cf89c7c55b7e2239b47201d704e8d2170884875b00f3103cf0662d6d7",
    strip_prefix = "zulu11.50.19-ca-jdk11.0.12-macosx_x64",
    urls = ["https://mirror.bazel.build/openjdk/azul-zulu11.50.19-ca-jdk11.0.12/zulu11.50.19-ca-jdk11.0.12-macosx_x64.tar.gz"],
)

# This must be kept in sync with src/test/shell/bazel/testdata/jdk_http_archives.
http_archive(
    name = "openjdk11_darwin_aarch64_archive",
    build_file_content = """
java_runtime(name = 'runtime', srcs =  glob(['**']), visibility = ['//visibility:public'])
exports_files(["WORKSPACE"], visibility = ["//visibility:public"])
""",
    sha256 = "e908a0b4c0da08d41c3e19230f819b364ff2e5f1dafd62d2cf991a85a34d3a17",
    strip_prefix = "zulu11.50.19-ca-jdk11.0.12-macosx_aarch64",
    urls = [
        "https://mirror.bazel.build/openjdk/azul-zulu11.50.19-ca-jdk11.0.12/zulu11.50.19-ca-jdk11.0.12-macosx_aarch64.tar.gz",
        "https://cdn.azul.com/zulu/bin/zulu11.50.19-ca-jdk11.0.12-macosx_aarch64.tar.gz",
    ],
)

# This must be kept in sync with src/test/shell/bazel/testdata/jdk_http_archives.
http_archive(
    name = "openjdk11_windows_archive",
    build_file_content = """
java_runtime(name = 'runtime', srcs =  glob(['**']), visibility = ['//visibility:public'])
exports_files(["WORKSPACE"], visibility = ["//visibility:public"])
""",
    sha256 = "42ae65e75d615a3f06a674978e1fa85fdf078cad94e553fee3e779b2b42bb015",
    strip_prefix = "zulu11.50.19-ca-jdk11.0.12-win_x64",
    urls = ["https://mirror.bazel.build/openjdk/azul-zulu11.50.19-ca-jdk11.0.12/zulu11.50.19-ca-jdk11.0.12-win_x64.zip"],
)

# This must be kept in sync with src/test/shell/bazel/testdata/jdk_http_archives.
http_archive(
    name = "openjdk17_linux_archive",
    build_file_content = """
java_runtime(name = 'runtime', srcs =  glob(['**']), visibility = ['//visibility:public'])
exports_files(["WORKSPACE"], visibility = ["//visibility:public"])
""",
    sha256 = "37c4f8e48536cceae8c6c20250d6c385e176972532fd35759fa7d6015c965f56",
    strip_prefix = "zulu17.28.13-ca-jdk17.0.0-linux_x64",
    urls = [
        "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu17.28.13-ca-jdk17.0.0-linux_x64.tar.gz",
        "https://cdn.azul.com/zulu/bin/zulu17.28.13-ca-jdk17.0.0-linux_x64.tar.gz",
    ],
)

# This must be kept in sync with src/test/shell/bazel/testdata/jdk_http_archives.
http_archive(
    name = "openjdk17_darwin_archive",
    build_file_content = """
java_runtime(name = 'runtime', srcs =  glob(['**']), visibility = ['//visibility:public'])
exports_files(["WORKSPACE"], visibility = ["//visibility:public"])
""",
    sha256 = "6029b1fe6853cecad22ab99ac0b3bb4fb8c903dd2edefa91c3abc89755bbd47d",
    strip_prefix = "zulu17.28.13-ca-jdk17.0.0-macosx_x64",
    urls = [
        "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu17.28.13-ca-jdk17.0.0-macosx_x64.tar.gz",
        "https://cdn.azul.com/zulu/bin/zulu17.28.13-ca-jdk17.0.0-macosx_x64.tar.gz",
    ],
)

# This must be kept in sync with src/test/shell/bazel/testdata/jdk_http_archives.
http_archive(
    name = "openjdk17_darwin_aarch64_archive",
    build_file_content = """
java_runtime(name = 'runtime', srcs =  glob(['**']), visibility = ['//visibility:public'])
exports_files(["WORKSPACE"], visibility = ["//visibility:public"])
""",
    sha256 = "6b17f01f767ee7abf4704149ca4d86423aab9b16b68697b7d36e9b616846a8b0",
    strip_prefix = "zulu17.28.13-ca-jdk17.0.0-macosx_aarch64",
    urls = [
        "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu17.28.13-ca-jdk17.0.0-macosx_aarch64.tar.gz",
        "https://cdn.azul.com/zulu/bin/zulu17.28.13-ca-jdk17.0.0-macosx_aarch64.tar.gz",
    ],
)

# This must be kept in sync with src/test/shell/bazel/testdata/jdk_http_archives.
http_archive(
    name = "openjdk17_windows_archive",
    build_file_content = """
java_runtime(name = 'runtime', srcs =  glob(['**']), visibility = ['//visibility:public'])
exports_files(["WORKSPACE"], visibility = ["//visibility:public"])
""",
    sha256 = "f4437011239f3f0031c794bb91c02a6350bc941d4196bdd19c9f157b491815a3",
    strip_prefix = "zulu17.28.13-ca-jdk17.0.0-win_x64",
    urls = [
        "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu17.28.13-ca-jdk17.0.0-win_x64.zip",
        "https://cdn.azul.com/zulu/bin/zulu17.28.13-ca-jdk17.0.0-win_x64.zip",
    ],
)

load("@io_bazel_skydoc//:setup.bzl", "stardoc_repositories")

stardoc_repositories()

load("@io_bazel_rules_sass//:package.bzl", "rules_sass_dependencies")

rules_sass_dependencies()

load("@build_bazel_rules_nodejs//:index.bzl", "node_repositories")

node_repositories()

load("@io_bazel_rules_sass//:defs.bzl", "sass_repositories")

sass_repositories()

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
