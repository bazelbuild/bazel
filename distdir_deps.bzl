# Copyright 2020 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""List the distribution dependencies we need to build Bazel.

Note for Bazel users: This is not the file that you are looking for.
This is internal source and is not intended to tell you what version
you should use for each dependency.
"""

DIST_DEPS = {
    ########################################
    #
    # Runtime language dependencies
    #
    ########################################
    "platforms": {
        "archive": "platforms-0.0.5.tar.gz",
        "sha256": "379113459b0feaf6bfbb584a91874c065078aa673222846ac765f86661c27407",
        "urls": [
            "https://mirror.bazel.build/github.com/bazelbuild/platforms/releases/download/0.0.5/platforms-0.0.5.tar.gz",
            "https://github.com/bazelbuild/platforms/releases/download/0.0.5/platforms-0.0.5.tar.gz",
        ],
        "used_in": [
            "additional_distfiles",
            "test_WORKSPACE_files",
        ],
    },
    "bazelci_rules": {
        "archive": "bazelci_rules-1.0.0.tar.gz",
        "sha256": "eca21884e6f66a88c358e580fd67a6b148d30ab57b1680f62a96c00f9bc6a07e",
        "strip_prefix": "bazelci_rules-1.0.0",
        "urls": [
            "https://mirror.bazel.build/github.com/bazelbuild/continuous-integration/releases/download/rules-1.0.0/bazelci_rules-1.0.0.tar.gz",
            "https://github.com/bazelbuild/continuous-integration/releases/download/rules-1.0.0/bazelci_rules-1.0.0.tar.gz",
        ],
        "used_in": [
            "additional_distfiles",
        ],
    },
    # Keep in sync with src/main/java/com/google/devtools/build/lib/bazel/rules/cpp/cc_configure.WORKSPACE.
    # Keep in sync with src/main/java/com/google/devtools/build/lib/bazel/rules/java/jdk.WORKSPACE.
    # Note: This is not in sync with src/test/java/com/google/devtools/build/lib/blackbox/framework/BlackBoxTestEnvironment.java.
    #       Perhaps it should be.
    "rules_cc": {
        "archive": "b1c40e1de81913a3c40e5948f78719c28152486d.zip",
        "sha256": "d0c573b94a6ef20ef6ff20154a23d0efcb409fb0e1ff0979cec318dfe42f0cdd",
        "strip_prefix": "rules_cc-b1c40e1de81913a3c40e5948f78719c28152486d",
        "urls": [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_cc/archive/b1c40e1de81913a3c40e5948f78719c28152486d.zip",
            "https://github.com/bazelbuild/rules_cc/archive/b1c40e1de81913a3c40e5948f78719c28152486d.zip",
        ],
        "used_in": [
            "additional_distfiles",
            "test_WORKSPACE_files",
        ],
    },
    "rules_java": {
        "archive": "7cf3cefd652008d0a64a419c34c13bdca6c8f178.zip",
        "sha256": "bc81f1ba47ef5cc68ad32225c3d0e70b8c6f6077663835438da8d5733f917598",
        "strip_prefix": "rules_java-7cf3cefd652008d0a64a419c34c13bdca6c8f178",
        "urls": [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_java/archive/7cf3cefd652008d0a64a419c34c13bdca6c8f178.zip",
            "https://github.com/bazelbuild/rules_java/archive/7cf3cefd652008d0a64a419c34c13bdca6c8f178.zip",
        ],
        "used_in": [
            "additional_distfiles",
            "test_WORKSPACE_files",
        ],
    },
    # TODO(aiuto): Update src/test/java/com/google/devtools/build/lib/blackbox/framework/BlackBoxTestEnvironment.java to use
    # this dynamically.
    "rules_proto": {
        "archive": "7e4afce6fe62dbff0a4a03450143146f9f2d7488.tar.gz",
        "sha256": "8e7d59a5b12b233be5652e3d29f42fba01c7cbab09f6b3a8d0a57ed6d1e9a0da",
        "strip_prefix": "rules_proto-7e4afce6fe62dbff0a4a03450143146f9f2d7488",
        "urls": [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_proto/archive/7e4afce6fe62dbff0a4a03450143146f9f2d7488.tar.gz",
            "https://github.com/bazelbuild/rules_proto/archive/7e4afce6fe62dbff0a4a03450143146f9f2d7488.tar.gz",
        ],
        "used_in": [
            "additional_distfiles",
            "test_WORKSPACE_files",
        ],
    },
    #################################################
    #
    # Dependencies which are part of the Bazel binary
    #
    #################################################
    "com_google_protobuf": {
        "archive": "v3.19.2.tar.gz",
        "sha256": "4dd35e788944b7686aac898f77df4e9a54da0ca694b8801bd6b2a9ffc1b3085e",
        "strip_prefix": "protobuf-3.19.2",
        "urls": [
            "https://mirror.bazel.build/github.com/protocolbuffers/protobuf/archive/v3.19.2.tar.gz",
            "https://github.com/protocolbuffers/protobuf/archive/v3.19.2.tar.gz",
        ],
        "patch_args": ["-p1"],
        "patches": ["//third_party/protobuf:3.19.2.patch"],
        "used_in": [
            "additional_distfiles",
            "test_WORKSPACE_files",
        ],
    },
    "protocolbuffers": {
        "archive": "2de300726a1ba2de9a468468dc5ff9ed17a3215f.tar.gz",
        "sha256": "6a5f67874af66b239b709c572ac1a5a00fdb1b29beaf13c3e6f79b1ba10dc7c4",
        "urls": [
            "https://mirror.bazel.build/github.com/protocolbuffers/upb/archive/2de300726a1ba2de9a468468dc5ff9ed17a3215f.tar.gz",
            "https://github.com/protocolbuffers/upb/archive/2de300726a1ba2de9a468468dc5ff9ed17a3215f.tar.gz",
        ],
        "used_in": [
            "additional_distfiles",
            "test_WORKSPACE_files",
        ],
    },
    "com_github_grpc_grpc": {
        "archive": "v1.41.0.tar.gz",
        "sha256": "e5fb30aae1fa1cffa4ce00aa0bbfab908c0b899fcf0bbc30e268367d660d8656",
        "strip_prefix": "grpc-1.41.0",
        "urls": [
            "https://mirror.bazel.build/github.com/grpc/grpc/archive/v1.41.0.tar.gz",
            "https://github.com/grpc/grpc/archive/v1.41.0.tar.gz",
        ],
        "patch_args": ["-p1"],
        "patches": [
            "//third_party/grpc:grpc_1.41.0.patch",
            "//third_party/grpc:grpc_1.41.0.win_arm64.patch",
        ],
        "used_in": [
            "additional_distfiles",
            "test_WORKSPACE_files",
        ],
    },
    "c-ares": {
        "archive": "e982924acee7f7313b4baa4ee5ec000c5e373c30.tar.gz",
        "sha256": "e8c2751ddc70fed9dc6f999acd92e232d5846f009ee1674f8aee81f19b2b915a",
        "urls": [
            "https://mirror.bazel.build/github.com/c-ares/c-ares/archive/e982924acee7f7313b4baa4ee5ec000c5e373c30.tar.gz",
            "https://github.com/c-ares/c-ares/archive/e982924acee7f7313b4baa4ee5ec000c5e373c30.tar.gz",
        ],
        "used_in": [
            "additional_distfiles",
            "test_WORKSPACE_files",
        ],
    },
    "re2": {
        "archive": "aecba11114cf1fac5497aeb844b6966106de3eb6.tar.gz",
        "sha256": "9f385e146410a8150b6f4cb1a57eab7ec806ced48d427554b1e754877ff26c3e",
        "urls": [
            "https://mirror.bazel.build/github.com/google/re2/archive/aecba11114cf1fac5497aeb844b6966106de3eb6.tar.gz",
            "https://github.com/google/re2/archive/aecba11114cf1fac5497aeb844b6966106de3eb6.tar.gz",
        ],
        "used_in": [
            "additional_distfiles",
            "test_WORKSPACE_files",
        ],
    },
    "abseil-cpp": {
        "archive": "997aaf3a28308eba1b9156aa35ab7bca9688e9f6.tar.gz",
        "sha256": "35f22ef5cb286f09954b7cc4c85b5a3f6221c9d4df6b8c4a1e9d399555b366ee",
        "urls": [
            "https://mirror.bazel.build/github.com/abseil/abseil-cpp/archive/997aaf3a28308eba1b9156aa35ab7bca9688e9f6.tar.gz",
            "https://github.com/abseil/abseil-cpp/archive/997aaf3a28308eba1b9156aa35ab7bca9688e9f6.tar.gz",
        ],
        "used_in": [
            "additional_distfiles",
            "test_WORKSPACE_files",
        ],
    },
    "zstd-jni": {
        "archive": "v1.5.0-4.zip",
        "patch_args": ["-p1"],
        "patches": [
            "//third_party:zstd-jni/Native.java.patch",
        ],
        "sha256": "d320d59b89a163c5efccbe4915ae6a49883ce653cdc670643dfa21c6063108e4",
        "urls": [
            "https://mirror.bazel.build/github.com/luben/zstd-jni/archive/v1.5.0-4.zip",
            "https://github.com/luben/zstd-jni/archive/v1.5.0-4.zip",
        ],
        "used_in": [
            "additional_distfiles",
        ],
    },
    ###################################################
    #
    # Build time dependencies for testing and packaging
    #
    ###################################################
    "bazel_skylib": {
        "archive": "bazel-skylib-1.0.3.tar.gz",
        "sha256": "1c531376ac7e5a180e0237938a2536de0c54d93f5c278634818e0efc952dd56c",
        "urls": [
            "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.0.3/bazel-skylib-1.0.3.tar.gz",
            "https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.3/bazel-skylib-1.0.3.tar.gz",
        ],
        "used_in": [
            "additional_distfiles",
            "test_WORKSPACE_files",
        ],
    },
    "io_bazel_skydoc": {
        "archive": "1ef781ced3b1443dca3ed05dec1989eca1a4e1cd.tar.gz",
        "sha256": "5a725b777976b77aa122b707d1b6f0f39b6020f66cd427bb111a585599c857b1",
        "urls": [
            "https://mirror.bazel.build/github.com/bazelbuild/stardoc/archive/1ef781ced3b1443dca3ed05dec1989eca1a4e1cd.tar.gz",
            "https://github.com/bazelbuild/stardoc/archive/1ef781ced3b1443dca3ed05dec1989eca1a4e1cd.tar.gz",
        ],
        "used_in": [
            "additional_distfiles",
        ],
        "strip_prefix": "stardoc-1ef781ced3b1443dca3ed05dec1989eca1a4e1cd",
    },
    "rules_pkg": {
        "archive": "rules_pkg-0.4.0.tar.gz",
        "sha256": "038f1caa773a7e35b3663865ffb003169c6a71dc995e39bf4815792f385d837d",
        "urls": [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_pkg/releases/download/0.4.0/rules_pkg-0.4.0.tar.gz",
            "https://github.com/bazelbuild/rules_pkg/releases/download/0.4.0/rules_pkg-0.4.0.tar.gz",
        ],
        "used_in": [
            "additional_distfiles",
        ],
    },
    # for Stardoc
    "io_bazel_rules_sass": {
        "archive": "1.25.0.zip",
        "sha256": "c78be58f5e0a29a04686b628cf54faaee0094322ae0ac99da5a8a8afca59a647",
        "strip_prefix": "rules_sass-1.25.0",
        "urls": [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_sass/archive/1.25.0.zip",
            "https://github.com/bazelbuild/rules_sass/archive/1.25.0.zip",
        ],
        "used_in": [
            "additional_distfiles",
        ],
    },
    # for Stardoc
    "build_bazel_rules_nodejs": {
        "archive": "rules_nodejs-2.2.2.tar.gz",
        "sha256": "f2194102720e662dbf193546585d705e645314319554c6ce7e47d8b59f459e9c",
        "urls": [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_nodejs/releases/download/2.2.2/rules_nodejs-2.2.2.tar.gz",
            "https://github.com/bazelbuild/rules_nodejs/releases/download/2.2.2/rules_nodejs-2.2.2.tar.gz",
        ],
        "used_in": [
            "additional_distfiles",
        ],
    },
    "desugar_jdk_libs": {
        # Commit 5847d6a06302136d95a14b4cbd4b55a9c9f1436e of 2021-03-10
        "archive": "5847d6a06302136d95a14b4cbd4b55a9c9f1436e.zip",
        "sha256": "299452e6f4a4981b2e6d22357f7332713382a63e4c137f5fd6b89579f6d610cb",
        "strip_prefix": "desugar_jdk_libs-5847d6a06302136d95a14b4cbd4b55a9c9f1436e",
        "urls": [
            "https://mirror.bazel.build/github.com/google/desugar_jdk_libs/archive/5847d6a06302136d95a14b4cbd4b55a9c9f1436e.zip",
            "https://github.com/google/desugar_jdk_libs/archive/5847d6a06302136d95a14b4cbd4b55a9c9f1436e.zip",
        ],
        "used_in": [
            "additional_distfiles",
        ],
    },
    "remote_coverage_tools": {
        "archive": "coverage_output_generator-v2.5.zip",
        "sha256": "cd14f1cb4559e4723e63b7e7b06d09fcc3bd7ba58d03f354cdff1439bd936a7d",
        "urls": [
            "https://mirror.bazel.build/bazel_coverage_output_generator/releases/coverage_output_generator-v2.5.zip",
        ],
        "used_in": [
            "additional_distfiles",
            "test_WORKSPACE_files",
        ],
    },
    "remote_java_tools": {
        "aliases": [
            "remote_java_tools_test",
            "remote_java_tools_for_testing",
        ],
        "archive": "java_tools-v11.6.zip",
        "sha256": "a7ac5922ee01e8b8fcb546ffc264ef314d0a0c679328b7fa4c432e5f54a86067",
        "urls": [
            "https://mirror.bazel.build/bazel_java_tools/releases/java/v11.6/java_tools-v11.6.zip",
            "https://github.com/bazelbuild/java_tools/releases/download/java_v11.6/java_tools-v11.6.zip",
        ],
        "used_in": [
            "additional_distfiles",
            "test_WORKSPACE_files",
        ],
    },
    "remote_java_tools_linux": {
        "aliases": [
            "remote_java_tools_test_linux",
            "remote_java_tools_linux_for_testing",
        ],
        "archive": "java_tools_linux-v11.6.zip",
        "sha256": "15da4f84a7d39cd179acf3035d9def638eea6ba89a0ed8f4e8a8e6e1d6c8e328",
        "urls": [
            "https://mirror.bazel.build/bazel_java_tools/releases/java/v11.6/java_tools_linux-v11.6.zip",
            "https://github.com/bazelbuild/java_tools/releases/download/java_v11.6/java_tools_linux-v11.6.zip",
        ],
        "used_in": [
            "additional_distfiles",
            "test_WORKSPACE_files",
        ],
    },
    "remote_java_tools_windows": {
        "aliases": [
            "remote_java_tools_test_windows",
            "remote_java_tools_windows_for_testing",
        ],
        "archive": "java_tools_windows-v11.6.zip",
        "sha256": "939f9d91f0df02851bbad8f5b1d26d24011329394cafe5668c1234e31ac2a1f7",
        "urls": [
            "https://mirror.bazel.build/bazel_java_tools/releases/java/v11.6/java_tools_windows-v11.6.zip",
            "https://github.com/bazelbuild/java_tools/releases/download/java_v11.6/java_tools_windows-v11.6.zip",
        ],
        "used_in": [
            "additional_distfiles",
            "test_WORKSPACE_files",
        ],
    },
    "remote_java_tools_darwin": {
        "aliases": [
            "remote_java_tools_test_darwin",
            "remote_java_tools_darwin_for_testing",
        ],
        "archive": "java_tools_darwin-v11.6.zip",
        "sha256": "f17ee54582b61f1ebd84c8fa2c54df796914cfbaac3cb821fb1286b55b080bc0",
        "urls": [
            "https://mirror.bazel.build/bazel_java_tools/releases/java/v11.6/java_tools_darwin-v11.6.zip",
            "https://github.com/bazelbuild/java_tools/releases/download/java_v11.6/java_tools_darwin-v11.6.zip",
        ],
        "used_in": [
            "additional_distfiles",
            "test_WORKSPACE_files",
        ],
    },
    "remotejdk11_linux": {
        "aliases": [
            "remotejdk11_linux_for_testing",
            "openjdk11_linux_archive",
            "openjdk_linux_vanilla",
        ],
        "archive": "zulu11.52.13-ca-jdk11.0.13-linux_x64.tar.gz",
        "sha256": "77a126669b26b3a89e0117b0f28cddfcd24fcd7699b2c1d35f921487148b9a9f",
        "strip_prefix": "zulu11.52.13-ca-jdk11.0.13-linux_x64",
        "urls": [
            "https://mirror.bazel.build/zulu/bin/zulu11.52.13-ca-jdk11.0.13-linux_x64.tar.gz",
            "https://cdn.azul.com/zulu/bin/zulu11.52.13-ca-jdk11.0.13-linux_x64.tar.gz",
        ],
        "used_in": ["test_WORKSPACE_files"],
    },
    "remotejdk11_linux_aarch64": {
        "aliases": [
            "remotejdk11_linux_aarch64_for_testing",
            "openjdk_linux_aarch64_vanilla",
        ],
        "archive": "zulu11.52.13-ca-jdk11.0.13-linux_aarch64.tar.gz",
        "sha256": "6be187379c26506a4b804b4f734c17e554aebe4204bde58a10b429054cc9cf9f",
        "strip_prefix": "zulu11.52.13-ca-jdk11.0.13-linux_aarch64",
        "urls": [
            "https://mirror.bazel.build/zulu-embedded/bin/zulu11.52.13-ca-jdk11.0.13-linux_aarch64.tar.gz",
            "https://cdn.azul.com/zulu-embedded/bin/zulu11.52.13-ca-jdk11.0.13-linux_aarch64.tar.gz",
        ],
        "used_in": ["test_WORKSPACE_files"],
    },
    "remotejdk11_linux_ppc64le": {
        "aliases": [
            "remotejdk11_linux_ppc64le_for_testing",
            "openjdk_linux_ppc64le_vanilla",
        ],
        "sha256": "82f14cda71cff99c878bf8400598a87235adb6c81b0337f7077c27e5cac1190c",
        "strip_prefix": "jdk-11.0.13+8",
        "urls": [
            "https://mirror.bazel.build/github.com/adoptium/temurin11-binaries/releases/download/jdk-11.0.13+8/OpenJDK11U-jdk_ppc64le_linux_hotspot_11.0.13_8.tar.gz",
            "https://github.com/adoptium/temurin11-binaries/releases/download/jdk-11.0.13+8/OpenJDK11U-jdk_ppc64le_linux_hotspot_11.0.13_8.tar.gz",
        ],
        "used_in": [],
    },
    "remotejdk11_linux_s390x": {
        "aliases": [
            "remotejdk11_linux_s390x_for_testing",
            "openjdk_linux_s390x_vanilla",
        ],
        "sha256": "9d280d86fdf6a7d9e5cbf54dc37f1d6d09dfe676ff5c684802fdfa3932eee63e",
        "strip_prefix": "jdk-11.0.13+8",
        "urls": [
            "https://mirror.bazel.build/github.com/adoptium/temurin11-binaries/releases/download/jdk-11.0.13+8/OpenJDK11U-jdk_s390x_linux_hotspot_11.0.13_8.tar.gz",
            "https://github.com/adoptium/temurin11-binaries/releases/download/jdk-11.0.13+8/OpenJDK11U-jdk_s390x_linux_hotspot_11.0.13_8.tar.gz",
        ],
        "used_in": [],
    },
    "remotejdk11_macos": {
        "aliases": [
            "remotejdk11_macos_for_testing",
            "openjdk_macos_x86_64_vanilla",
            "openjdk11_darwin_archive",
        ],
        "archive": "zulu11.52.13-ca-jdk11.0.13-macosx_x64.tar.gz",
        "sha256": "e27a11a6e970ba6f597ecc957c0cdb502ff8990c243a6abd9df1e3413a0a3e44",
        "strip_prefix": "zulu11.52.13-ca-jdk11.0.13-macosx_x64",
        "urls": [
            "https://mirror.bazel.build/zulu/bin/zulu11.52.13-ca-jdk11.0.13-macosx_x64.tar.gz",
            "https://cdn.azul.com/zulu/bin/zulu11.52.13-ca-jdk11.0.13-macosx_x64.tar.gz",
        ],
        "used_in": ["test_WORKSPACE_files"],
    },
    "remotejdk11_macos_aarch64": {
        "aliases": [
            "openjdk_macos_aarch64_vanilla",
            "remotejdk11_macos_aarch64_for_testing",
            "openjdk11_darwin_aarch64_archive",
        ],
        "archive": "zulu11.52.13-ca-jdk11.0.13-macosx_aarch64.tar.gz",
        "sha256": "eb8d593a61a9461a554e7bb1d67bca0f94242273f1d01a13f58c20c269b35fe5",
        "strip_prefix": "zulu11.52.13-ca-jdk11.0.13-macosx_aarch64",
        "urls": [
            "https://mirror.bazel.build/zulu/bin/zulu11.52.13-ca-jdk11.0.13-macosx_aarch64.tar.gz",
            "https://cdn.azul.com/zulu/bin/zulu11.52.13-ca-jdk11.0.13-macosx_aarch64.tar.gz",
        ],
        "used_in": ["test_WORKSPACE_files"],
    },
    "remotejdk11_win": {
        "aliases": [
            "remotejdk11_win_for_testing",
            "openjdk11_windows_archive",
            "openjdk_win_vanilla",
        ],
        "archive": "zulu11.52.13-ca-jdk11.0.13-win_x64.zip",
        "sha256": "49289cc181504a01c946d5a66a5c904f8e6d912a2226dd82db62eb37e6c0a9d9",
        "strip_prefix": "zulu11.52.13-ca-jdk11.0.13-win_x64",
        "urls": [
            "https://mirror.bazel.build/zulu/bin/zulu11.52.13-ca-jdk11.0.13-win_x64.zip",
            "https://cdn.azul.com/zulu/bin/zulu11.52.13-ca-jdk11.0.13-win_x64.zip",
        ],
        "used_in": ["test_WORKSPACE_files"],
    },
    "remotejdk17_win_arm64": {
        "aliases": [
            "remotejdk17_win_arm64_for_testing",
            "openjdk17_windows_arm64_archive",
            "openjdk_win_arm64_vanilla",
        ],
        "archive": "zulu17.30.15-ca-jdk17.0.1-win_aarch64.zip",
        "sha256": "811d7e7591bac4f081dfb00ba6bd15b6fc5969e1f89f0f327ef75147027c3877",
        "strip_prefix": "zulu17.30.15-ca-jdk17.0.1-win_aarch64",
        "urls": [
            "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu17.30.15-ca-jdk17.0.1-win_aarch64.zip",
            "https://cdn.azul.com/zulu/bin/zulu17.30.15-ca-jdk17.0.1-win_aarch64.zip",
        ],
        "used_in": ["test_WORKSPACE_files"],
    },
    "remotejdk11_win_arm64": {
        "aliases": [
            "remotejdk11_win_arm64_for_testing",
            "openjdk11_windows_arm64_archive",
        ],
        "archive": "microsoft-jdk-11.0.13.8.1-windows-aarch64.zip",
        "sha256": "b8a28e6e767d90acf793ea6f5bed0bb595ba0ba5ebdf8b99f395266161e53ec2",
        "strip_prefix": "jdk-11.0.13+8",
        "urls": [
            "https://mirror.bazel.build/aka.ms/download-jdk/microsoft-jdk-11.0.13.8.1-windows-aarch64.zip",
        ],
        "used_in": ["test_WORKSPACE_files"],
    },
    "remotejdk17_linux": {
        "aliases": [
            "remotejdk17_linux_for_testing",
            "openjdk17_linux_archive",
        ],
        "sha256": "73d5c4bae20325ca41b606f7eae64669db3aac638c5b3ead4a975055846ad6de",
        "strip_prefix": "zulu17.32.13-ca-jdk17.0.2-linux_x64",
        "urls": [
            "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu17.32.13-ca-jdk17.0.2-linux_x64.tar.gz",
            "https://cdn.azul.com/zulu/bin/zulu17.32.13-ca-jdk17.0.2-linux_x64.tar.gz",
        ],
        "used_in": [],
    },
    "remotejdk17_macos": {
        "aliases": [
            "remotejdk17_macos_for_testing",
            "openjdk17_darwin_archive",
        ],
        "sha256": "89d04b2d99b05dcb25114178e65f6a1c5ca742e125cab0a63d87e7e42f3fcb80",
        "strip_prefix": "zulu17.32.13-ca-jdk17.0.2-macosx_x64",
        "urls": [
            "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu17.32.13-ca-jdk17.0.2-macosx_x64.tar.gz",
            "https://cdn.azul.com/zulu/bin/zulu17.32.13-ca-jdk17.0.2-macosx_x64.tar.gz",
        ],
        "used_in": [],
    },
    "remotejdk17_macos_aarch64": {
        "aliases": [
            "remotejdk17_macos_aarch64_for_testing",
            "openjdk17_darwin_aarch64_archive",
        ],
        "sha256": "54247dde248ffbcd3c048675504b1c503b81daf2dc0d64a79e353c48d383c977",
        "strip_prefix": "zulu17.32.13-ca-jdk17.0.2-macosx_aarch64",
        "urls": [
            "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu17.32.13-ca-jdk17.0.2-macosx_aarch64.tar.gz",
            "https://cdn.azul.com/zulu/bin/zulu17.32.13-ca-jdk17.0.2-macosx_aarch64.tar.gz",
        ],
        "used_in": [],
    },
    "remotejdk17_win": {
        "aliases": [
            "remotejdk17_win_for_testing",
            "openjdk17_windows_archive",
        ],
        "sha256": "e965aa0ea7a0661a3446cf8f10ee00684b851f883b803315289f26b4aa907fdb",
        "strip_prefix": "zulu17.32.13-ca-jdk17.0.2-win_x64",
        "urls": [
            "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu17.32.13-ca-jdk17.0.2-win_x64.zip",
            "https://cdn.azul.com/zulu/bin/zulu17.32.13-ca-jdk17.0.2-win_x64.zip",
        ],
        "used_in": [],
    },
}

# Add aliased names
DEPS_BY_NAME = {}

def _create_index():
    for repo_name in DIST_DEPS:
        repo = DIST_DEPS[repo_name]
        DEPS_BY_NAME[repo_name] = repo
        aliases = repo.get("aliases")
        if aliases:
            for alias in aliases:
                DEPS_BY_NAME[alias] = repo

_create_index()

def _gen_workspace_stanza_impl(ctx):
    if ctx.attr.template and (ctx.attr.preamble or ctx.attr.postamble):
        fail("Can not use template with either preamble or postamble")
    if ctx.attr.use_maybe and ctx.attr.repo_clause:
        fail("Can not use use_maybe with repo_clause")

    if ctx.attr.use_maybe:
        repo_clause = """
maybe(
    http_archive,
    name = "{repo}",
    sha256 = "{sha256}",
    strip_prefix = {strip_prefix},
    urls = {urls},
)
"""
    elif ctx.attr.repo_clause:
        repo_clause = ctx.attr.repo_clause
    else:
        repo_clause = """
http_archive(
    name = "{repo}",
    sha256 = "{sha256}",
    strip_prefix = {strip_prefix},
    urls = {urls},
)
"""

    repo_stanzas = {}
    for repo in ctx.attr.repos:
        info = DEPS_BY_NAME[repo]
        strip_prefix = info.get("strip_prefix")
        if strip_prefix:
            strip_prefix = "\"%s\"" % strip_prefix
        else:
            strip_prefix = "None"

        repo_stanzas["{%s}" % repo] = repo_clause.format(
            repo = repo,
            sha256 = str(info["sha256"]),
            strip_prefix = strip_prefix,
            urls = info["urls"],
        )

    if ctx.attr.template:
        ctx.actions.expand_template(
            output = ctx.outputs.out,
            template = ctx.file.template,
            substitutions = repo_stanzas,
        )
    else:
        content = "\n".join([p.strip() for p in ctx.attr.preamble.strip().split("\n")])
        content += "\n"
        content += "".join(repo_stanzas.values())
        content += "\n"
        content += "\n".join([p.strip() for p in ctx.attr.postamble.strip().split("\n")])
        content += "\n"
        ctx.actions.write(ctx.outputs.out, content)

    return [DefaultInfo(files = depset([ctx.outputs.out]))]

gen_workspace_stanza = rule(
    attrs = {
        "repos": attr.string_list(doc = "Set of repos to include."),
        "out": attr.output(mandatory = True),
        "preamble": attr.string(doc = "Preamble."),
        "postamble": attr.string(doc = "Set of rules to follow repos."),
        "template": attr.label(
            doc = "Template WORKSPACE file. May not be used with preamble or postamble." +
                  "Repo stanzas can be included using the syntax '{repo name}'.",
            allow_single_file = True,
            mandatory = False,
        ),
        "use_maybe": attr.bool(doc = "Use maybe() invocation instead of http_archive."),
        "repo_clause": attr.string(doc = "Use a custom clause for each repository."),
    },
    doc = "Use specifications from DIST_DEPS to generate WORKSPACE http_archive stanzas or to" +
          "drop them into a template.",
    implementation = _gen_workspace_stanza_impl,
)
