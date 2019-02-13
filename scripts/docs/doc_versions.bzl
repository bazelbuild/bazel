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

# To get the checksum of the versioned documentation tree tarball, run the
# following command with the selected Bazel version:
#
# $ curl -s https://mirror.bazel.build/bazel_versioned_docs/jekyll-tree-0.20.0.tar | sha256sum | cut -d" " -f1
# bb79a63810bf1b0aa1f89bd3bbbeb4a547a30ab9af70c9be656cc6866f4b015b
#
# This list must be kept in sync with `doc_versions` variable in //site:_config.yml
"""This module contains the versions and hashes of Bazel's documentation tarballs."""

DOC_VERSIONS = [
    {
        "version": "0.22.0",
        "sha256": "bec5cfaa5560e082e41e33bde276cf93f0f7bcfd2914a3e868f921df8b3ab725",
    },
    {
        "version": "0.21.0",
        "sha256": "23ec39c0138d358c544151e5c81586716d5d1c6124f10a742bead70516e6eb93",
    },
    {
        "version": "0.20.0",
        "sha256": "bb79a63810bf1b0aa1f89bd3bbbeb4a547a30ab9af70c9be656cc6866f4b015b",
    },
    {
        "version": "0.19.2",
        "sha256": "3c2d9f21ec2fd1c0b8a310f6eb6043027c838810cdfc2457d4346a0e5cdcaa7a",
    },
    {
        "version": "0.19.1",
        "sha256": "ec892c59ba18bb8de1f9ae2bde937db144e45f28d6d1c32a2cee847ee81b134d",
    },
    {
        "version": "0.18.1",
        "sha256": "98b77f48e37a50fc6f83100bf53f661e10732bb3ddbc226e02d0225cb7a9a7d8",
    },
    {
        "version": "0.17.2",
        "sha256": "13b35dd309a0d52f0a2518a1193f42729c75255f5fae40cea68e4d4224bfaa2e",
    },
    {
        "version": "0.17.1",
        "sha256": "02256ddd20eeaf70cf8fcfe9b2cdddd7be87aedd5848d549474fb0358e0031d3",
    },
]
