# Copyright 2020 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""List the distribution dependencies we need to build Bazel."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def dist_http_archive(name, **kwargs):
    """Wraps http_archive but takes sha and urls from DIST_DEPS.

    dist_http_archive wraps an http_archive invocation, but looks up relevant
    information from DIST_DEPS so the user does not have to specify it. It
    always strips sha256 and urls from kwargs.

    Args:
      name: repo name
      **kwargs: see http_archive for allowed args.
    """
    info = DIST_DEPS[name]
    if "patches" not in kwargs:
        kwargs["patches"] = info.get("patches")
    if "strip_prefix" not in kwargs:
        kwargs["strip_prefix"] = info.get("strip_prefix")
    http_archive(
        name = name,
        sha256 = info["sha256"],
        urls = info["urls"],
        **kwargs
    )

DIST_DEPS = {
    ########################################
    #
    # Runtime language dependencies
    #
    ########################################
    "platforms": {
        "archive": "platforms-0.0.2.tar.gz",
        "sha256": "48a2d8d343863989c232843e01afc8a986eb8738766bfd8611420a7db8f6f0c3",
        "urls": [
            "https://mirror.bazel.build/github.com/bazelbuild/platforms/releases/download/0.0.2/platforms-0.0.2.tar.gz",
            "https://github.com/bazelbuild/platforms/releases/download/0.0.2/platforms-0.0.2.tar.gz"
        ],
        "used_in": [
            "additional_distfiles",
            "test_WORKSPACE_files"
        ],
    },
    "bazel_toolchains": {
        "archive": "bazel-toolchains-3.1.0.tar.gz",
        "sha256": "726b5423e1c7a3866a3a6d68e7123b4a955e9fcbe912a51e0f737e6dab1d0af2",
        "urls": [
            "https://mirror.bazel.build/github.com/bazelbuild/bazel-toolchains/releases/download/3.1.0/bazel-toolchains-3.1.0.tar.gz",
            "https://github.com/bazelbuild/bazel-toolchains/releases/download/3.1.0/bazel-toolchains-3.1.0.tar.gz"
        ],
        "used_in": [
            "additional_distfiles"
        ],
        "strip_prefix": "bazel-toolchains-3.1.0"
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
            "test_WORKSPACE_files"
        ],
    },
    #################################################
    #
    # Dependencies which are part of the Bazel binary
    #
    #################################################
    "com_google_protobuf": {
        "archive": "v3.13.0.tar.gz",
        "sha256": "9b4ee22c250fe31b16f1a24d61467e40780a3fbb9b91c3b65be2a376ed913a1a",
        "strip_prefix": "protobuf-3.13.0",
        "urls": [
            "https://mirror.bazel.build/github.com/protocolbuffers/protobuf/archive/v3.13.0.tar.gz",
            "https://github.com/protocolbuffers/protobuf/archive/v3.13.0.tar.gz"
        ],
        "used_in": [
            "additional_distfiles",
            "test_WORKSPACE_files"
        ],
    },
    "c-ares": {
        "archive": "e982924acee7f7313b4baa4ee5ec000c5e373c30.tar.gz",
        "sha256": "e8c2751ddc70fed9dc6f999acd92e232d5846f009ee1674f8aee81f19b2b915a",
        "urls": [
            "https://mirror.bazel.build/github.com/c-ares/c-ares/archive/e982924acee7f7313b4baa4ee5ec000c5e373c30.tar.gz",
            "https://github.com/c-ares/c-ares/archive/e982924acee7f7313b4baa4ee5ec000c5e373c30.tar.gz"
        ],
        "used_in": [
            "additional_distfiles",
            "test_WORKSPACE_files"
        ],
    },
    "re2": {
        "archive": "aecba11114cf1fac5497aeb844b6966106de3eb6.tar.gz",
        "sha256": "9f385e146410a8150b6f4cb1a57eab7ec806ced48d427554b1e754877ff26c3e",
        "urls": [
            "https://mirror.bazel.build/github.com/google/re2/archive/aecba11114cf1fac5497aeb844b6966106de3eb6.tar.gz",
            "https://github.com/google/re2/archive/aecba11114cf1fac5497aeb844b6966106de3eb6.tar.gz"
        ],
        "used_in": [
            "additional_distfiles",
            "test_WORKSPACE_files"
        ],
    },
    "abseil-cpp": {
        "archive": "df3ea785d8c30a9503321a3d35ee7d35808f190d.tar.gz",
        "sha256": "f368a8476f4e2e0eccf8a7318b98dafbe30b2600f4e3cf52636e5eb145aba06a",
        "urls": [
            "https://mirror.bazel.build/github.com/abseil/abseil-cpp/archive/df3ea785d8c30a9503321a3d35ee7d35808f190d.tar.gz",
            "https://github.com/abseil/abseil-cpp/archive/df3ea785d8c30a9503321a3d35ee7d35808f190d.tar.gz"
        ],
        "used_in": [
            "additional_distfiles",
            "test_WORKSPACE_files"
        ],
    },
    ###################################################
    #
    # Build time dependencies for testing and packaging
    #
    ###################################################
    "rules_pkg": {
        "archive": "rules_pkg-0.2.4.tar.gz",
        "sha256": "4ba8f4ab0ff85f2484287ab06c0d871dcb31cc54d439457d28fd4ae14b18450a",
        "urls": [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_pkg/releases/download/0.2.4/rules_pkg-0.2.4.tar.gz",
            "https://github.com/bazelbuild/rules_pkg/releases/download/0.2.4/rules_pkg-0.2.4.tar.gz",
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
}
