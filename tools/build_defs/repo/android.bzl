# Copyright 2018 The Bazel Authors. All rights reserved.
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

# WARNING:
# https://github.com/bazelbuild/bazel/issues/17713
# .bzl files in this package (tools/build_defs/repo) are evaluated
# in a Starlark environment without "@_builtins" injection, and must not refer
# to symbols associated with build/workspace .bzl files

"""Rules for importing external Android Archives (AARs).

Usage:

    # In WORKSPACE
    load("@bazel_tools//tools/build_defs/repo:android.bzl", "aar_import_external", "aar_maven_import_external")

    # Specify the URL directly:
    aar_import_external(
        name = "com_android_support_preference_v14_25_1_0",                              # required
        licenses = ["notice"],                                                           # required
        aar_urls = [                                                                     # required
            "https://dl.google.com/dl/android/maven2/com/android/support/preference-v14/25.1.0/preference-v14-25.1.0.aar"
        ],
        aar_sha256 = "442473fe5c395ebef26c14eb01d17ceda33ad207a4cc23a32a2ad95b87edfabb", # optional or empty string
        deps = [                                                                         # optional or empty list
            "@com_android_support_recyclerview_v7_25_1_0//aar",
            "@com_android_support_appcompat_v7_25_1_0//aar",
            "@com_android_support_preference_v7_25_1_0//aar",
            "@com_android_support_support_v4_25_1_0//aar",
        ],
    )

    # Or, specify the artifact coordinate:
    aar_maven_import_external(
        name = "com_android_support_preference_v14_25_1_0",                         # required
        artifact = "com.android.support.test:preference-v14:25.1.0",                # required
        sha256 = "442473fe5c395ebef26c14eb01d17ceda33ad207a4cc23a32a2ad95b87edfabb" # optional or empty string
        licenses = ["notice"],                                                      # required
        server_urls = ["https://maven.google.com"],                                 # required
        deps = [                                                                    # optional or empty list
            "@com_android_support_recyclerview_v7_25_1_0//aar",
            "@com_android_support_appcompat_v7_25_1_0//aar",
            "@com_android_support_preference_v7_25_1_0//aar",
            "@com_android_support_support_v4_25_1_0//aar",
        ],
    )

    # In BUILD.bazel
    android_library(
        name = "foo",
        srcs = [...],
        deps = [
            "@com_android_support_preference_v14_25_1_0//aar",
        ],
    )
"""

load(":jvm.bzl", "convert_artifact_coordinate_to_urls", "jvm_import_external")

def aar_import_external(aar_sha256, aar_urls, **kwargs):
    jvm_import_external(
        rule_name = "aar_import",
        rule_metadata = {
            "extension": "aar",
            "import_attr": "aar = %s",
        },
        artifact_sha256 = aar_sha256,
        artifact_urls = aar_urls,
        **kwargs
    )

def aar_maven_import_external(artifact, server_urls, aar_sha256 = "", **kwargs):
    aar_import_external(
        aar_sha256 = aar_sha256,
        aar_urls = convert_artifact_coordinate_to_urls(
            artifact,
            server_urls,
            "aar",
        ),
        **kwargs
    )
