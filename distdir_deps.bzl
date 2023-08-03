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
        "package_version": "1.0.0",
    },
    # Used in src/main/java/com/google/devtools/build/lib/bazel/rules/cpp/cc_configure.WORKSPACE.
    # Used in src/main/java/com/google/devtools/build/lib/bazel/rules/java/jdk.WORKSPACE.
    # Used in src/test/java/com/google/devtools/build/lib/blackbox/framework/blackbox.WORKSAPCE
    "rules_cc": {
        "archive": "rules_cc-0.0.8.tar.gz",
        "sha256": "ae46b722a8b8e9b62170f83bfb040cbf12adb732144e689985a66b26410a7d6f",
        "urls": ["https://github.com/bazelbuild/rules_cc/releases/download/0.0.8/rules_cc-0.0.8.tar.gz"],
        "used_in": [
            "additional_distfiles",
        ],
        "package_version": "0.0.8",
        "strip_prefix": "rules_cc-0.0.8",
    },
    "rules_java": {
        "aliases": [
            "rules_java_builtin",
            "rules_java_builtin_for_testing",
        ],
        "archive": "rules_java-6.3.0.tar.gz",
        "sha256": "29ba147c583aaf5d211686029842c5278e12aaea86f66bd4a9eb5e525b7f2701",
        "urls": ["https://github.com/bazelbuild/rules_java/releases/download/6.3.0/rules_java-6.3.0.tar.gz"],
        "workspace_file_content": "",
        "used_in": [
            "additional_distfiles",
        ],
        "license_kinds": [
            "@rules_license//licenses/spdx:Apache-2.0",
        ],
        "package_version": "6.1.1",
    },
    # Used in src/test/java/com/google/devtools/build/lib/blackbox/framework/blackbox.WORKSAPCE
    "rules_proto": {
        "archive": "5.3.0-21.7.tar.gz",
        "sha256": "dc3fb206a2cb3441b485eb1e423165b231235a1ea9b031b4433cf7bc1fa460dd",
        "strip_prefix": "rules_proto-5.3.0-21.7",
        "urls": [
            "https://github.com/bazelbuild/rules_proto/archive/refs/tags/5.3.0-21.7.tar.gz",
        ],
        "used_in": [
            "additional_distfiles",
        ],
        "license_kinds": [
            "@rules_license//licenses/spdx:Apache-2.0",
        ],
    },
    #################################################
    #
    # Dependencies which are part of the Bazel binary
    #
    #################################################
    "com_google_protobuf": {
        "archive": "v21.7.tar.gz",
        "sha256": "75be42bd736f4df6d702a0e4e4d30de9ee40eac024c4b845d17ae4cc831fe4ae",
        "strip_prefix": "protobuf-21.7",
        "urls": [
            "https://mirror.bazel.build/github.com/protocolbuffers/protobuf/archive/v21.7.tar.gz",
            "https://github.com/protocolbuffers/protobuf/archive/v21.7.tar.gz",
        ],
        "patch_args": ["-p1"],
        "patches": ["//third_party/protobuf:21.7.patch"],
        "used_in": [
            "additional_distfiles",
        ],
        "license_kinds": [
            "@rules_license//licenses/generic:notice",
        ],
        "license_text": "LICENSE",
        "package_version": "3.19.6",
    },
    "com_github_grpc_grpc": {
        "archive": "v1.48.1.tar.gz",
        "sha256": "320366665d19027cda87b2368c03939006a37e0388bfd1091c8d2a96fbc93bd8",
        "strip_prefix": "grpc-1.48.1",
        "urls": [
            "https://mirror.bazel.build/github.com/grpc/grpc/archive/v1.48.1.tar.gz",
            "https://github.com/grpc/grpc/archive/v1.48.1.tar.gz",
        ],
        "patch_args": ["-p1"],
        "patches": [
            "//third_party/grpc:grpc_1.48.1.patch",
            "//third_party/grpc:grpc_1.48.1.win_arm64.patch",
        ],
        "used_in": [
            "additional_distfiles",
        ],
        "package_version": "1.48.1",
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
        ],
        "package_version": "0.24.0",
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
        ],
        "license_kinds": [
            "@rules_license//licenses/spdx:Apache-2.0",
        ],
        "license_text": "LICENSE",
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
        ],
        "license_kinds": [
            "@rules_license//licenses/generic:notice",
        ],
        "license_text": "LICENSE",
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
        ],
    },
    "com_google_absl": {
        "archive": "20220623.1.tar.gz",
        "sha256": "91ac87d30cc6d79f9ab974c51874a704de9c2647c40f6932597329a282217ba8",
        "urls": [
            "https://mirror.bazel.build/github.com/abseil/abseil-cpp/archive/refs/tags/20220623.1.tar.gz",
            "https://github.com/abseil/abseil-cpp/archive/refs/tags/20220623.1.tar.gz",
        ],
        "used_in": [
            "additional_distfiles",
        ],
        "strip_prefix": "abseil-cpp-20220623.1",
        "license_kinds": [
            "@rules_license//licenses/generic:notice",
        ],
        "license_text": "LICENSE",
        "package_version": "20220623.1",
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
        "license_kinds": [
            "@rules_license//licenses/spdx:BSD-2-Clause",
        ],
        "license_text": "LICENSE",
        "package_version": "1.5.2-3",
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
        ],
        "package_version": "8.0.40",
    },
    "bazel_skylib": {
        "archive": "bazel-skylib-1.3.0.tar.gz",
        "sha256": "74d544d96f4a5bb630d465ca8bbcfe231e3594e5aae57e1edbf17a6eb3ca2506",
        "urls": [
            "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.3.0/bazel-skylib-1.3.0.tar.gz",
            "https://github.com/bazelbuild/bazel-skylib/releases/download/1.3.0/bazel-skylib-1.3.0.tar.gz",
        ],
        "used_in": [
            "additional_distfiles",
        ],
        "package_version": "1.3.0",
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
        "archive": "rules_license-0.0.7.tar.gz",
        "sha256": "4531deccb913639c30e5c7512a054d5d875698daeb75d8cf90f284375fe7c360",
        "urls": [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_license/releases/download/0.0.7/rules_license-0.0.7.tar.gz",
            "https://github.com/bazelbuild/rules_license/releases/download/0.0.7/rules_license-0.0.7.tar.gz",
        ],
        "used_in": [
            "additional_distfiles",
        ],
        "package_version": "0.0.7",
    },
    "rules_pkg": {
        "archive": "rules_pkg-0.8.0.tar.gz",
        "sha256": "eea0f59c28a9241156a47d7a8e32db9122f3d50b505fae0f33de6ce4d9b61834",
        "urls": [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_pkg/releases/download/0.8.0/rules_pkg-0.8.0.tar.gz",
            "https://github.com/bazelbuild/rules_pkg/releases/download/0.8.0/rules_pkg-0.8.0.tar.gz",
        ],
        "used_in": [
            "additional_distfiles",
        ],
        "package_version": "0.8.0",
    },
    "rules_jvm_external": {
        "archive": "rules_jvm_external-5.2.tar.gz",
        "sha256": "f86fd42a809e1871ca0aabe89db0d440451219c3ce46c58da240c7dcdc00125f",
        "strip_prefix": "rules_jvm_external-5.2",
        "patches": [
            "//third_party:rules_jvm_external_5.2.patch",
        ],
        "patch_args": ["-p1"],
        "urls": [
            "https://github.com/bazelbuild/rules_jvm_external/releases/download/5.2/rules_jvm_external-5.2.tar.gz",
        ],
        "used_in": [
            "additional_distfiles",
        ],
        "package_version": "5.2",
    },
    "rules_python": {
        "sha256": "ffc7b877c95413c82bfd5482c017edcf759a6250d8b24e82f41f3c8b8d9e287e",
        "strip_prefix": "rules_python-0.19.0",
        "urls": ["https://github.com/bazelbuild/rules_python/releases/download/0.19.0/rules_python-0.19.0.tar.gz"],
        "archive": "rules_python-0.19.0.tar.gz",
        "used_in": ["additional_distfiles"],
    },
    "rules_testing": {
        "sha256": "4e21f9aa7996944ce91431f27bca374bff56e680acfe497276074d56bc5d9af2",
        "strip_prefix": "rules_testing-0.0.4",
        "urls": [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_testing/releases/download/v0.0.4/rules_testing-v0.0.4.tar.gz",
            "https://github.com/bazelbuild/rules_testing/releases/download/v0.0.4/rules_testing-v0.0.4.tar.gz",
        ],
        "archive": "rules_testing-v0.0.4.tar.gz",
        "used_in": ["additional_distfiles"],
        "package_version": "0.0.4",
    },
    "desugar_jdk_libs": {
        # Commit 5847d6a06302136d95a14b4cbd4b55a9c9f1436e of 2021-03-10
        "archive": "5847d6a06302136d95a14b4cbd4b55a9c9f1436e.zip",
        "sha256": "299452e6f4a4981b2e6d22357f7332713382a63e4c137f5fd6b89579f6d610cb",
        "strip_prefix": "desugar_jdk_libs-5847d6a06302136d95a14b4cbd4b55a9c9f1436e",
        "patch_args": ["-p1"],
        "patches": ["//third_party:desugar_jdk_libs-exec_tools.patch"],
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
        ],
        "package_version": "2.6",
    },
    "openjdk_linux_vanilla": {
        "archive": "zulu20.30.11-ca-jdk20.0.1-linux_x64.tar.gz",
        "sha256": "ec5c0426a0eb2b0460968a044665ed4603b224acd5e20c379e9d7890511da683",
        "strip_prefix": "zulu20.30.11-ca-jdk20.0.1-linux_x64",
        "urls": [
            "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu20.30.11-ca-jdk20.0.1-linux_x64.tar.gz",
            "https://cdn.azul.com/zulu/bin/zulu20.30.11-ca-jdk20.0.1-linux_x64.tar.gz",
        ],
        "used_in": [
        ],
    },
    "openjdk_linux_aarch64_vanilla": {
        "archive": "zulu20.30.11-ca-jdk20.0.1-linux_aarch64.tar.gz",
        "sha256": "2487cf315d1f56291c1f41fb56a34a7f863ce5bf85cadd284c79ea3f848d707c",
        "strip_prefix": "zulu20.30.11-ca-jdk20.0.1-linux_aarch64",
        "urls": [
            "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu20.30.11-ca-jdk20.0.1-linux_aarch64.tar.gz",
            "https://cdn.azul.com/zulu/bin/zulu20.30.11-ca-jdk20.0.1-linux_aarch64.tar.gz",
        ],
        "used_in": [
        ],
    },
    # JDK20 unavailable so use JDK19 instead for linux s390x.
    "openjdk_linux_s390x_vanilla": {
        "archive": "OpenJDK19U-jdk_s390x_linux_hotspot_19.0.2_7.tar.gz",
        "sha256": "f2512f9a8e9847dd5d3557c39b485a8e7a1ef37b601dcbcb748d22e49f44815c",
        "strip_prefix": "jdk-19.0.2+7",
        "urls": [
            "https://mirror.bazel.build/github.com/adoptium/temurin19-binaries/releases/download/jdk-19.0.2%2B7/OpenJDK19U-jdk_s390x_linux_hotspot_19.0.2_7.tar.gz",
            "https://github.com/adoptium/temurin19-binaries/releases/download/jdk-19.0.2%2B7/OpenJDK19U-jdk_s390x_linux_hotspot_19.0.2_7.tar.gz",
        ],
        "used_in": [
        ],
    },
    "openjdk_linux_ppc64le_vanilla": {
        "archive": "OpenJDK20U-jdk_ppc64le_linux_hotspot_20_36.tar.gz",
        "sha256": "45dde71faf8cbb78fab3c976894259655c8d3de827347f23e0ebe5710921dded",
        "strip_prefix": "jdk-20+36",
        "urls": [
            "https://mirror.bazel.build/github.com/adoptium/temurin20-binaries/releases/download/jdk-20%2B36/OpenJDK20U-jdk_ppc64le_linux_hotspot_20_36.tar.gz",
            "https://github.com/adoptium/temurin20-binaries/releases/download/jdk-20%2B36/OpenJDK20U-jdk_ppc64le_linux_hotspot_20_36.tar.gz",
        ],
        "used_in": [],
    },
    "openjdk_macos_x86_64_vanilla": {
        "archive": "zulu20.30.11-ca-jdk20.0.1-macosx_x64.tar.gz",
        "sha256": "befee9db92345d5146945061b721d3a6c6e182471c1536f87dbadfd5aab0e241",
        "strip_prefix": "zulu20.30.11-ca-jdk20.0.1-macosx_x64",
        "urls": [
            "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu20.30.11-ca-jdk20.0.1-macosx_x64.tar.gz",
            "https://cdn.azul.com/zulu/bin/zulu20.30.11-ca-jdk20.0.1-macosx_x64.tar.gz",
        ],
        "used_in": [
        ],
    },
    "openjdk_macos_aarch64_vanilla": {
        "archive": "zulu20.30.11-ca-jdk20.0.1-macosx_aarch64.tar.gz",
        "sha256": "01e59f0160d051524bb16d865652d25d00a85390581737a8f35f89057c80892d",
        "strip_prefix": "zulu20.30.11-ca-jdk20.0.1-macosx_aarch64",
        "urls": [
            "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu20.30.11-ca-jdk20.0.1-macosx_aarch64.tar.gz",
            "https://cdn.azul.com/zulu/bin/zulu20.30.11-ca-jdk20.0.1-macosx_aarch64.tar.gz",
        ],
        "used_in": [
        ],
    },
    "openjdk_win_vanilla": {
        "archive": "zulu20.30.11-ca-jdk20.0.1-win_x64.zip",
        "sha256": "8a97ee11da578292f7c9e772f3edd3f083fa4f34f47a98e3abefb625ab2225ba",
        "strip_prefix": "zulu20.30.11-ca-jdk20.0.1-win_x64",
        "urls": [
            "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu20.30.11-ca-jdk20.0.1-win_x64.zip",
            "https://cdn.azul.com/zulu/bin/zulu20.30.11-ca-jdk20.0.1-win_x64.zip",
        ],
        "used_in": [
        ],
    },
    # JDK20 unavailable so use JDK19 instead for win aarch64.
    "openjdk_win_arm64_vanilla": {
        "archive": "zulu19.28.81-ca-jdk19.0.0-win_aarch64.zip",
        "sha256": "e73e851638066c48421a60e01ce7d956c1de0935620e1b66d8bbbd6cdd4f815e",
        "strip_prefix": "zulu19.28.81-ca-jdk19.0.0-win_aarch64",
        "urls": [
            "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu19.28.81-ca-jdk19.0.0-win_aarch64.zip",
            "https://cdn.azul.com/zulu/bin/zulu19.28.81-ca-jdk19.0.0-win_aarch64.zip",
        ],
        "used_in": [
        ],
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
