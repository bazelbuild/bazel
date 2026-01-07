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
        "archive": "platforms-0.0.7.tar.gz",
        "sha256": "3a561c99e7bdbe9173aa653fd579fe849f1d8d67395780ab4770b1f381431d51",
        "urls": [
            "https://mirror.bazel.build/github.com/bazelbuild/platforms/releases/download/0.0.7/platforms-0.0.7.tar.gz",
            "https://github.com/bazelbuild/platforms/releases/download/0.0.7/platforms-0.0.7.tar.gz",
        ],
        "used_in": [
            "additional_distfiles",
            "test_WORKSPACE_files",
        ],
        "package_version": "0.0.7",
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
    # Used in src/main/java/com/google/devtools/build/lib/bazel/rules/cpp/cc_configure.WORKSPACE.
    # Used in src/main/java/com/google/devtools/build/lib/bazel/rules/java/jdk.WORKSPACE.
    # Used in src/test/java/com/google/devtools/build/lib/blackbox/framework/blackbox.WORKSAPCE
    "rules_cc": {
        "archive": "rules_cc-0.0.2.tar.gz",
        "sha256": "58bff40957ace85c2de21ebfc72e53ed3a0d33af8cc20abd0ceec55c63be7de2",
        "urls": ["https://github.com/bazelbuild/rules_cc/releases/download/0.0.2/rules_cc-0.0.2.tar.gz"],
        "used_in": [
            "additional_distfiles",
            "test_WORKSPACE_files",
        ],
    },
    "rules_java": {
        "archive": "rules_java-5.5.1.tar.gz",
        "sha256": "73b88f34dc251bce7bc6c472eb386a6c2b312ed5b473c81fe46855c248f792e0",
        "strip_prefix": "",
        "urls": [
            "https://github.com/bazelbuild/rules_java/releases/download/5.5.1/rules_java-5.5.1.tar.gz",
        ],
        "used_in": [
            "additional_distfiles",
            "test_WORKSPACE_files",
        ],
    },
    # Used in src/test/java/com/google/devtools/build/lib/blackbox/framework/blackbox.WORKSAPCE
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
        "license_kinds": [
            "@rules_license//licenses/spdx:Apache-2.0",
        ],
        "package_version": "2020-10-27",
    },
    #################################################
    #
    # Dependencies which are part of the Bazel binary
    #
    #################################################
    "com_google_protobuf": {
        "archive": "v3.19.6.tar.gz",
        "sha256": "9a301cf94a8ddcb380b901e7aac852780b826595075577bb967004050c835056",
        "strip_prefix": "protobuf-3.19.6",
        "urls": [
            "https://mirror.bazel.build/github.com/protocolbuffers/protobuf/archive/v3.19.6.tar.gz",
            "https://github.com/protocolbuffers/protobuf/archive/v3.19.6.tar.gz",
        ],
        "patch_args": ["-p1"],
        "patches": ["//third_party/protobuf:3.19.6.patch"],
        "used_in": [
            "additional_distfiles",
            "test_WORKSPACE_files",
        ],
    },
    "com_github_grpc_grpc": {
        "archive": "v1.47.0.tar.gz",
        "sha256": "271bdc890bf329a8de5b65819f0f9590a5381402429bca37625b63546ed19e54",
        "strip_prefix": "grpc-1.47.0",
        "urls": [
            "https://mirror.bazel.build/github.com/grpc/grpc/archive/v1.47.0.tar.gz",
            "https://github.com/grpc/grpc/archive/v1.47.0.tar.gz",
        ],
        "patch_args": ["-p1"],
        "patches": [
            "//third_party/grpc:grpc_1.47.0.patch",
            "//third_party/grpc:grpc_1.47.0.win_arm64.patch",
        ],
        "used_in": [
            "additional_distfiles",
            "test_WORKSPACE_files",
        ],
    },
    "com_github_cncf_udpa": {
        "archive": "cb28da3451f158a947dfc45090fe92b07b243bc1.tar.gz",
        "sha256": "5bc8365613fe2f8ce6cc33959b7667b13b7fe56cb9d16ba740c06e1a7c4242fc",
        "urls": [
            "https://mirror.bazel.build/github.com/cncf/xds/archive/cb28da3451f158a947dfc45090fe92b07b243bc1.tar.gz",
            "https://github.com/cncf/xds/archive/cb28da3451f158a947dfc45090fe92b07b243bc1.tar.gz",
        ],
        "strip_prefix": "xds-cb28da3451f158a947dfc45090fe92b07b243bc1",
        "patch_args": ["-p1"],
        "patches": [
            "//third_party/cncf_udpa:cncf_udpa_0.0.1.patch",
        ],
        "used_in": [
            "additional_distfiles",
            "test_WORKSPACE_files",
        ],
    },
    "com_envoyproxy_protoc_gen_validate": {
        "archive": "4694024279bdac52b77e22dc87808bd0fd732b69.tar.gz",
        "sha256": "1e490b98005664d149b379a9529a6aa05932b8a11b76b4cd86f3d22d76346f47",
        "strip_prefix": "protoc-gen-validate-4694024279bdac52b77e22dc87808bd0fd732b69",
        "urls": [
            "https://mirror.bazel.build/github.com/envoyproxy/protoc-gen-validate/archive/4694024279bdac52b77e22dc87808bd0fd732b69.tar.gz",
            "https://github.com/envoyproxy/protoc-gen-validate/archive/4694024279bdac52b77e22dc87808bd0fd732b69.tar.gz",
        ],
        "patch_args": ["-p1"],
        "patches": [
            "//third_party/protoc_gen_validate:protoc_gen_validate.patch",
        ],
        "used_in": [
            "additional_distfiles",
            "test_WORKSPACE_files",
        ],
    },
    "bazel_gazelle": {
        "archive": "bazel-gazelle-v0.24.0.tar.gz",
        "sha256": "de69a09dc70417580aabf20a28619bb3ef60d038470c7cf8442fafcf627c21cb",
        "urls": [
            "https://mirror.bazel.build/github.com/bazelbuild/bazel-gazelle/releases/download/v0.24.0/bazel-gazelle-v0.24.0.tar.gz",
            "https://github.com/bazelbuild/bazel-gazelle/releases/download/v0.24.0/bazel-gazelle-v0.24.0.tar.gz",
        ],
        "used_in": [
            "additional_distfiles",
            "test_WORKSPACE_files",
        ],
    },
    "com_google_googleapis": {
        "archive": "2f9af297c84c55c8b871ba4495e01ade42476c92.tar.gz",
        "sha256": "5bb6b0253ccf64b53d6c7249625a7e3f6c3bc6402abd52d3778bfa48258703a0",
        "strip_prefix": "googleapis-2f9af297c84c55c8b871ba4495e01ade42476c92",
        "urls": [
            "https://mirror.bazel.build/github.com/googleapis/googleapis/archive/2f9af297c84c55c8b871ba4495e01ade42476c92.tar.gz",
            "https://github.com/googleapis/googleapis/archive/2f9af297c84c55c8b871ba4495e01ade42476c92.tar.gz",
        ],
        "used_in": [
            "additional_distfiles",
            "test_WORKSPACE_files",
        ],
    },
    "upb": {
        "archive": "a5477045acaa34586420942098f5fecd3570f577.tar.gz",
        "sha256": "cf7f71eaff90b24c1a28b49645a9ff03a9a6c1e7134291ce70901cb63e7364b5",
        "strip_prefix": "upb-a5477045acaa34586420942098f5fecd3570f577",
        "urls": [
            "https://mirror.bazel.build/github.com/protocolbuffers/upb/archive/a5477045acaa34586420942098f5fecd3570f577.tar.gz",
            "https://github.com/protocolbuffers/upb/archive/a5477045acaa34586420942098f5fecd3570f577.tar.gz",
        ],
        "used_in": [
            "additional_distfiles",
            "test_WORKSPACE_files",
        ],
        "license_kinds": [
            "@rules_license//licenses/generic:notice",
        ],
        "patch_args": ["-p1"],
        "patches": [
            "//third_party/upb:01_remove_werror.patch",
        ],
    },
    "c-ares": {
        "archive": "6654436a307a5a686b008c1d4c93b0085da6e6d8.tar.gz",
        "sha256": "ec76c5e79db59762776bece58b69507d095856c37b81fd35bfb0958e74b61d93",
        "urls": [
            "https://mirror.bazel.build/github.com/c-ares/c-ares/archive/6654436a307a5a686b008c1d4c93b0085da6e6d8.tar.gz",
            "https://github.com/c-ares/c-ares/archive/6654436a307a5a686b008c1d4c93b0085da6e6d8.tar.gz",
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
    "com_google_absl": {
        "archive": "20230802.0.tar.gz",
        "sha256": "59d2976af9d6ecf001a81a35749a6e551a335b949d34918cfade07737b9d93c5",
        "urls": [
            "https://github.com/abseil/abseil-cpp/archive/refs/tags/20230802.0.tar.gz",
        ],
        "used_in": [
            "additional_distfiles",
            "test_WORKSPACE_files",
        ],
        "strip_prefix": "abseil-cpp-20230802.0",
    },
    "zstd-jni": {
        "archive": "v1.5.2-3.zip",
        "patch_args": ["-p1"],
        "patches": [
            "//third_party:zstd-jni/Native.java.patch",
        ],
        "sha256": "366009a43cfada35015e4cc40a7efc4b7f017c6b8df5cac3f87d2478027b2056",
        "urls": [
            "https://mirror.bazel.build/github.com/luben/zstd-jni/archive/refs/tags/v1.5.2-3.zip",
            "https://github.com/luben/zstd-jni/archive/refs/tags/v1.5.2-3.zip",
        ],
        "used_in": [
            "additional_distfiles",
        ],
    },
    "blake3": {
        "archive": "1.3.3.zip",
        "sha256": "bb529ba133c0256df49139bd403c17835edbf60d2ecd6463549c6a5fe279364d",
        "urls": [
            "https://github.com/BLAKE3-team/BLAKE3/archive/refs/tags/1.3.3.zip",
        ],
        "used_in": [
            "additional_distfiles",
        ],
        "license_kinds": [
            "@rules_license//licenses/spdx:Apache-2.0",
        ],
        "license_text": "LICENSE",
        "package_version": "1.3.3",
    },
    ###################################################
    #
    # Build time dependencies for testing and packaging
    #
    ###################################################
    "android_gmaven_r8": {
        "archive": "r8-8.0.40.jar",
        "sha256": "ab1379835c7d3e5f21f80347c3c81e2f762e0b9b02748ae5232c3afa14adf702",
        "urls": [
            "https://maven.google.com/com/android/tools/r8/8.0.40/r8-8.0.40.jar",
        ],
        "used_in": [
            "test_WORKSPACE_files",
        ],
        "package_version": "8.0.40",
    },
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
    "rules_license": {
        "archive": "rules_license-0.0.3.tar.gz",
        "sha256": "00ccc0df21312c127ac4b12880ab0f9a26c1cff99442dc6c5a331750360de3c3",
        "urls": [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_license/releases/download/0.0.3/rules_license-0.0.3.tar.gz",
            "https://github.com/bazelbuild/rules_license/releases/download/0.0.3/rules_license-0.0.3.tar.gz",
        ],
        "used_in": [
            "additional_distfiles",
            "test_WORKSPACE_files",
        ],
    },
    "rules_pkg": {
        "archive": "rules_pkg-0.7.0.tar.gz",
        "sha256": "8a298e832762eda1830597d64fe7db58178aa84cd5926d76d5b744d6558941c2",
        "urls": [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_pkg/releases/download/0.7.0/rules_pkg-0.7.0.tar.gz",
            "https://github.com/bazelbuild/rules_pkg/releases/download/0.7.0/rules_pkg-0.7.0.tar.gz",
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
    "build_bazel_rules_nodejs": {
        "archive": "rules_nodejs-5.5.0.tar.gz",
        "sha256": "0fad45a9bda7dc1990c47b002fd64f55041ea751fafc00cd34efb96107675778",
        "urls": [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_nodejs/releases/download/5.5.0/rules_nodejs-5.5.0.tar.gz",
            "https://github.com/bazelbuild/rules_nodejs/releases/download/5.5.0/rules_nodejs-5.5.0.tar.gz",
        ],
        "used_in": [
            "additional_distfiles",
        ],
    },
    "rules_nodejs": {
        "archive": "rules_nodejs-core-5.5.0.tar.gz",
        "sha256": "4d48998e3fa1e03c684e6bdf7ac98051232c7486bfa412e5b5475bbaec7bb257",
        "urls": [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_nodejs/releases/download/5.5.0/rules_nodejs-core-5.5.0.tar.gz",
            "https://github.com/bazelbuild/rules_nodejs/releases/download/5.5.0/rules_nodejs-core-5.5.0.tar.gz",
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
        "archive": "coverage_output_generator-v2.6.zip",
        "sha256": "7006375f6756819b7013ca875eab70a541cf7d89142d9c511ed78ea4fefa38af",
        "urls": [
            "https://mirror.bazel.build/bazel_coverage_output_generator/releases/coverage_output_generator-v2.6.zip",
        ],
        "used_in": [
            "test_WORKSPACE_files",
        ],
    },
    "remote_java_tools": {
        "aliases": [
            "remote_java_tools_test",
            "remote_java_tools_for_testing",
        ],
        "archive": "java_tools-v12.7.zip",
        "sha256": "aa11ecd5fc0af2769f0f2bdd25e2f4de7c1291ed24326fb23fa69bdd5dcae2b5",
        "urls": [
            "https://mirror.bazel.build/bazel_java_tools/releases/java/v12.7/java_tools-v12.7.zip",
            "https://github.com/bazelbuild/java_tools/releases/download/java_v12.7/java_tools-v12.7.zip",
        ],
        "used_in": [
            "test_WORKSPACE_files",
        ],
        "package_version": "12.7",
    },
    "remote_java_tools_linux": {
        "aliases": [
            "remote_java_tools_test_linux",
            "remote_java_tools_linux_for_testing",
        ],
        "archive": "java_tools_linux-v12.7.zip",
        "sha256": "a346b9a291b6db1bb06f7955f267e47522d99963fe14e337da1d75d125a8599f",
        "urls": [
            "https://mirror.bazel.build/bazel_java_tools/releases/java/v12.7/java_tools_linux-v12.7.zip",
            "https://github.com/bazelbuild/java_tools/releases/download/java_v12.7/java_tools_linux-v12.7.zip",
        ],
        "used_in": [
            "test_WORKSPACE_files",
        ],
        "package_version": "12.7",
    },
    "remote_java_tools_windows": {
        "aliases": [
            "remote_java_tools_test_windows",
            "remote_java_tools_windows_for_testing",
        ],
        "archive": "java_tools_windows-v12.7.zip",
        "sha256": "bae6a03b5aeead5804ba7bcdcc8b14ec3ed05b37f3db5519f788ab060bc53b05",
        "urls": [
            "https://mirror.bazel.build/bazel_java_tools/releases/java/v12.7/java_tools_windows-v12.7.zip",
            "https://github.com/bazelbuild/java_tools/releases/download/java_v12.7/java_tools_windows-v12.7.zip",
        ],
        "used_in": [
            "test_WORKSPACE_files",
        ],
        "package_version": "12.7",
    },
    "remote_java_tools_darwin_x86_64": {
        "aliases": [
            "remote_java_tools_test_darwin_x86_64",
            "remote_java_tools_darwin_x86_64_for_testing",
        ],
        "archive": "java_tools_darwin_x86_64-v12.7.zip",
        "sha256": "e116c649c0355ab57ffcc870ce1139e5e1528cabac458bd50263d2b84ea4ffb2",
        "urls": [
            "https://mirror.bazel.build/bazel_java_tools/releases/java/v12.7/java_tools_darwin_x86_64-v12.7.zip",
            "https://github.com/bazelbuild/java_tools/releases/download/java_v12.7/java_tools_darwin_x86_64-v12.7.zip",
        ],
        "used_in": [
            "test_WORKSPACE_files",
        ],
        "package_version": "12.7",
    },
    "remote_java_tools_darwin_arm64": {
        "aliases": [
            "remote_java_tools_test_darwin_arm64",
            "remote_java_tools_darwin_arm64_for_testing",
        ],
        "archive": "java_tools_darwin_arm64-v12.7.zip",
        "sha256": "ecedf6305768dfd51751d0ad732898af092bd7710d497c6c6c3214af7e49395f",
        "urls": [
            "https://mirror.bazel.build/bazel_java_tools/releases/java/v12.7/java_tools_darwin_arm64-v12.7.zip",
            "https://github.com/bazelbuild/java_tools/releases/download/java_v12.7/java_tools_darwin_arm64-v12.7.zip",
        ],
        "used_in": [
            "test_WORKSPACE_files",
        ],
        "package_version": "12.7",
    },
    "remotejdk11_linux": {
        "aliases": [
            "remotejdk11_linux_for_testing",
            "openjdk11_linux_archive",
            "openjdk_linux_vanilla",
        ],
        "archive": "zulu11.56.19-ca-jdk11.0.15-linux_x64.tar.gz",
        "sha256": "e064b61d93304012351242bf0823c6a2e41d9e28add7ea7f05378b7243d34247",
        "strip_prefix": "zulu11.56.19-ca-jdk11.0.15-linux_x64",
        "urls": [
            "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu11.56.19-ca-jdk11.0.15-linux_x64.tar.gz",
            "https://cdn.azul.com/zulu/bin/zulu11.56.19-ca-jdk11.0.15-linux_x64.tar.gz",
        ],
        "used_in": ["test_WORKSPACE_files"],
    },
    "remotejdk11_linux_aarch64": {
        "aliases": [
            "remotejdk11_linux_aarch64_for_testing",
            "openjdk_linux_aarch64_vanilla",
        ],
        "archive": "zulu11.56.19-ca-jdk11.0.15-linux_aarch64.tar.gz",
        "sha256": "fc7c41a0005180d4ca471c90d01e049469e0614cf774566d4cf383caa29d1a97",
        "strip_prefix": "zulu11.56.19-ca-jdk11.0.15-linux_aarch64",
        "urls": [
            "https://mirror.bazel.build/cdn.azul.com/zulu-embedded/bin/zulu11.56.19-ca-jdk11.0.15-linux_aarch64.tar.gz",
            "https://cdn.azul.com/zulu-embedded/bin/zulu11.56.19-ca-jdk11.0.15-linux_aarch64.tar.gz",
        ],
        "used_in": ["test_WORKSPACE_files"],
    },
    "remotejdk11_linux_ppc64le": {
        "aliases": [
            "remotejdk11_linux_ppc64le_for_testing",
            "openjdk_linux_ppc64le_vanilla",
        ],
        "sha256": "a8fba686f6eb8ae1d1a9566821dbd5a85a1108b96ad857fdbac5c1e4649fc56f",
        "strip_prefix": "jdk-11.0.15+10",
        "urls": [
            "https://mirror.bazel.build/github.com/adoptium/temurin11-binaries/releases/download/jdk-11.0.15+10/OpenJDK11U-jdk_ppc64le_linux_hotspot_11.0.15_10.tar.gz",
            "https://github.com/adoptium/temurin11-binaries/releases/download/jdk-11.0.15+10/OpenJDK11U-jdk_ppc64le_linux_hotspot_11.0.15_10.tar.gz",
        ],
        "used_in": [],
    },
    "remotejdk11_linux_s390x": {
        "aliases": [
            "remotejdk11_linux_s390x_for_testing",
            "openjdk_linux_s390x_vanilla",
        ],
        "sha256": "a58fc0361966af0a5d5a31a2d8a208e3c9bb0f54f345596fd80b99ea9a39788b",
        "strip_prefix": "jdk-11.0.15+10",
        "urls": [
            "https://mirror.bazel.build/github.com/adoptium/temurin11-binaries/releases/download/jdk-11.0.15+10/OpenJDK11U-jdk_s390x_linux_hotspot_11.0.15_10.tar.gz",
            "https://github.com/adoptium/temurin11-binaries/releases/download/jdk-11.0.15+10/OpenJDK11U-jdk_s390x_linux_hotspot_11.0.15_10.tar.gz",
        ],
        "used_in": [],
    },
    "remotejdk11_macos": {
        "aliases": [
            "remotejdk11_macos_for_testing",
            "openjdk_macos_x86_64_vanilla",
            "openjdk11_darwin_archive",
        ],
        "archive": "zulu11.56.19-ca-jdk11.0.15-macosx_x64.tar.gz",
        "sha256": "2614e5c5de8e989d4d81759de4c333aa5b867b17ab9ee78754309ba65c7f6f55",
        "strip_prefix": "zulu11.56.19-ca-jdk11.0.15-macosx_x64",
        "urls": [
            "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu11.56.19-ca-jdk11.0.15-macosx_x64.tar.gz",
            "https://cdn.azul.com/zulu/bin/zulu11.56.19-ca-jdk11.0.15-macosx_x64.tar.gz",
        ],
        "used_in": ["test_WORKSPACE_files"],
    },
    "remotejdk11_macos_aarch64": {
        "aliases": [
            "openjdk_macos_aarch64_vanilla",
            "remotejdk11_macos_aarch64_for_testing",
            "openjdk11_darwin_aarch64_archive",
        ],
        "archive": "zulu11.56.19-ca-jdk11.0.15-macosx_aarch64.tar.gz",
        "sha256": "6bb0d2c6e8a29dcd9c577bbb2986352ba12481a9549ac2c0bcfd00ed60e538d2",
        "strip_prefix": "zulu11.56.19-ca-jdk11.0.15-macosx_aarch64",
        "urls": [
            "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu11.56.19-ca-jdk11.0.15-macosx_aarch64.tar.gz",
            "https://cdn.azul.com/zulu/bin/zulu11.56.19-ca-jdk11.0.15-macosx_aarch64.tar.gz",
        ],
        "used_in": ["test_WORKSPACE_files"],
    },
    "remotejdk11_win": {
        "aliases": [
            "remotejdk11_win_for_testing",
            "openjdk11_windows_archive",
            "openjdk_win_vanilla",
        ],
        "archive": "zulu11.56.19-ca-jdk11.0.15-win_x64.zip",
        "sha256": "a106c77389a63b6bd963a087d5f01171bd32aa3ee7377ecef87531390dcb9050",
        "strip_prefix": "zulu11.56.19-ca-jdk11.0.15-win_x64",
        "urls": [
            "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu11.56.19-ca-jdk11.0.15-win_x64.zip",
            "https://cdn.azul.com/zulu/bin/zulu11.56.19-ca-jdk11.0.15-win_x64.zip",
        ],
        "used_in": ["test_WORKSPACE_files"],
    },
    "remotejdk17_win_arm64": {
        "aliases": [
            "remotejdk17_win_arm64_for_testing",
            "openjdk17_windows_arm64_archive",
            "openjdk_win_arm64_vanilla",
        ],
        "archive": "zulu17.38.21-ca-jdk17.0.5-win_aarch64.zip",
        "sha256": "bc3476f2161bf99bc9a243ff535b8fc033b34ce9a2fa4b62fb8d79b6bfdc427f",
        "strip_prefix": "zulu17.38.21-ca-jdk17.0.5-win_aarch64",
        "urls": [
            "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu17.38.21-ca-jdk17.0.5-win_aarch64.zip",
            "https://cdn.azul.com/zulu/bin/zulu17.38.21-ca-jdk17.0.5-win_aarch64.zip",
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
        "archive": "zulu17.38.21-ca-jdk17.0.5-linux_x64.tar.gz",
        "sha256": "20c91a922eec795f3181eaa70def8b99d8eac56047c9a14bfb257c85b991df1b",
        "strip_prefix": "zulu17.38.21-ca-jdk17.0.5-linux_x64",
        "urls": [
            "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu17.38.21-ca-jdk17.0.5-linux_x64.tar.gz",
            "https://cdn.azul.com/zulu/bin/zulu17.38.21-ca-jdk17.0.5-linux_x64.tar.gz",
        ],
        "used_in": [],
    },
    "remotejdk17_linux_aarch64": {
        "aliases": [
            "remotejdk17_linux_aarch64_for_testing",
            "openjdk17_linux_aarch64_archive",
        ],
        "archive": "zulu17.38.21-ca-jdk17.0.5-linux_aarch64.tar.gz",
        "sha256": "dbc6ae9163e7ff469a9ab1f342cd1bc1f4c1fb78afc3c4f2228ee3b32c4f3e43",
        "strip_prefix": "zulu17.38.21-ca-jdk17.0.5-linux_aarch64",
        "urls": [
            "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu17.38.21-ca-jdk17.0.5-linux_aarch64.tar.gz",
            "https://cdn.azul.com/zulu/bin/zulu17.38.21-ca-jdk17.0.5-linux_aarch64.tar.gz",
        ],
        "used_in": ["test_WORKSPACE_files"],
    },
    "remotejdk17_macos": {
        "aliases": [
            "remotejdk17_macos_for_testing",
            "openjdk17_darwin_archive",
        ],
        "archive": "zulu17.38.21-ca-jdk17.0.5-macosx_x64.tar.gz",
        "sha256": "e6317cee4d40995f0da5b702af3f04a6af2bbd55febf67927696987d11113b53",
        "strip_prefix": "zulu17.38.21-ca-jdk17.0.5-macosx_x64",
        "urls": [
            "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu17.38.21-ca-jdk17.0.5-macosx_x64.tar.gz",
            "https://cdn.azul.com/zulu/bin/zulu17.38.21-ca-jdk17.0.5-macosx_x64.tar.gz",
        ],
        "used_in": [],
    },
    "remotejdk17_macos_aarch64": {
        "aliases": [
            "remotejdk17_macos_aarch64_for_testing",
            "openjdk17_darwin_aarch64_archive",
        ],
        "archive": "zulu17.38.21-ca-jdk17.0.5-macosx_aarch64",
        "sha256": "515dd56ec99bb5ae8966621a2088aadfbe72631818ffbba6e4387b7ee292ab09",
        "strip_prefix": "zulu17.38.21-ca-jdk17.0.5-macosx_aarch64",
        "urls": [
            "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu17.38.21-ca-jdk17.0.5-macosx_aarch64.tar.gz",
            "https://cdn.azul.com/zulu/bin/zulu17.38.21-ca-jdk17.0.5-macosx_aarch64.tar.gz",
        ],
        "used_in": [],
    },
    "remotejdk17_win": {
        "aliases": [
            "remotejdk17_win_for_testing",
            "openjdk17_windows_archive",
        ],
        "archive": "zulu17.38.21-ca-jdk17.0.5-win_x64.zip",
        "sha256": "9972c5b62a61b45785d3d956c559e079d9e91f144ec46225f5deeda214d48f27",
        "strip_prefix": "zulu17.38.21-ca-jdk17.0.5-win_x64",
        "urls": [
            "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu17.38.21-ca-jdk17.0.5-win_x64.zip",
            "https://cdn.azul.com/zulu/bin/zulu17.38.21-ca-jdk17.0.5-win_x64.zip",
        ],
        "used_in": [],
    },
    "remotejdk18_linux_aarch64": {
        "aliases": [
            "remotejdk18_linux_for_testing",
            "openjdk18_linux_archive",
        ],
        "archive": "zulu18.28.13-ca-jdk18.0.0-linux_aarch64.tar.gz",
        "sha256": "a1d5f78172f32f819d08e9043b0f82fa7af738b37c55c6ca8d6092c61d204d53",
        "strip_prefix": "zulu18.28.13-ca-jdk18.0.0-linux_aarch64",
        "urls": [
            "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu18.28.13-ca-jdk18.0.0-linux_aarch64.tar.gz",
            "https://cdn.azul.com/zulu/bin/zulu18.28.13-ca-jdk18.0.0-linux_aarch64.tar.gz",
        ],
        "used_in": ["test_WORKSPACE_files"],
    },
    "remotejdk18_linux": {
        "aliases": [
            "remotejdk18_linux_for_testing",
            "openjdk18_linux_archive",
        ],
        "sha256": "959a94ca4097dcaabc7886784cec10dfdf2b0a3bff890ea8943cc09c5fff29cb",
        "strip_prefix": "zulu18.28.13-ca-jdk18.0.0-linux_x64",
        "urls": [
            "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu18.28.13-ca-jdk18.0.0-linux_x64.tar.gz",
            "https://cdn.azul.com/zulu/bin/zulu18.28.13-ca-jdk18.0.0-linux_x64.tar.gz",
        ],
        "used_in": [],
    },
    "remotejdk18_macos": {
        "aliases": [
            "remotejdk18_macos_for_testing",
            "openjdk18_darwin_archive",
        ],
        "sha256": "780a9aa4bda95a6793bf41d13f837c59ef915e9bfd0e0c5fd4c70e4cdaa88541",
        "strip_prefix": "zulu18.28.13-ca-jdk18.0.0-macosx_x64",
        "urls": [
            "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu18.28.13-ca-jdk18.0.0-macosx_x64.tar.gz",
            "https://cdn.azul.com/zulu/bin/zulu18.28.13-ca-jdk18.0.0-macosx_x64.tar.gz",
        ],
        "used_in": [],
    },
    "remotejdk18_macos_aarch64": {
        "aliases": [
            "remotejdk18_macos_aarch64_for_testing",
            "openjdk18_darwin_aarch64_archive",
        ],
        "sha256": "9595e001451e201fdf33c1952777968a3ac18fe37273bdeaea5b5ed2c4950432",
        "strip_prefix": "zulu18.28.13-ca-jdk18.0.0-macosx_aarch64",
        "urls": [
            "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu18.28.13-ca-jdk18.0.0-macosx_aarch64.tar.gz",
            "https://cdn.azul.com/zulu/bin/zulu18.28.13-ca-jdk18.0.0-macosx_aarch64.tar.gz",
        ],
        "used_in": [],
    },
    "remotejdk18_win": {
        "aliases": [
            "remotejdk18_win_for_testing",
            "openjdk18_windows_archive",
        ],
        "sha256": "6c75498163b047595386fdb909cb6d4e04282c3a81799743c5e1f9316391fe16",
        "strip_prefix": "zulu18.28.13-ca-jdk18.0.0-win_x64",
        "urls": [
            "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu18.28.13-ca-jdk18.0.0-win_x64.zip",
            "https://cdn.azul.com/zulu/bin/zulu18.28.13-ca-jdk18.0.0-win_x64.zip",
        ],
        "used_in": [],
    },
    "remotejdk18_win_arm64": {
        "aliases": [
            "remotejdk18_win_arm64_for_testing",
            "openjdk18_windows_arm64_archive",
        ],
        "archive": "zulu18.28.13-ca-jdk18.0.0-win_aarch64.zip",
        "sha256": "9b52b259516e4140ee56b91f77750667bffbc543e78ad8c39082449d4c377b54",
        "strip_prefix": "zulu18.28.13-ca-jdk18.0.0-win_aarch64",
        "urls": [
            "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu18.28.13-ca-jdk18.0.0-win_aarch64.zip",
            "https://cdn.azul.com/zulu/bin/zulu18.28.13-ca-jdk18.0.0-win_aarch64.zip",
        ],
        "used_in": ["test_WORKSPACE_files"],
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
