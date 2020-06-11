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
        "version": "3.2.0",
        "sha256": "6cff8654e739a0c3062183a5a6cc82fcf9a77323051f8c007866d7f4101052a6",
    },
    {
        "version": "3.1.0-807b377",  # cherrypicked 807b377.
        "sha256": "f9d2e22e24af426d6c9de163d91abe6d8af7eb1eabb1d7ff5e9cf4bededf465a",
    },
    {
        "version": "3.0.0",
        "sha256": "bd1096ad609c253fa7b1473edf4a3aa51f36243e188dbb62c68d8ed4aca2419d",
    },
    {
        "version": "2.2.0",
        "sha256": "4c1506786ab98df8039ec7354b82da7b586b2ae4ab7f7e7d08f3caf74ff28e3d",
    },
    {
        "version": "2.1.0",
        "sha256": "b0fd257b1d6b1b05705742d55a13b9a20d3e99f49c89334750c872d620e5b88f",
    },
    {
        "version": "2.0.0",
        "sha256": "7d7c424ede503856c61b645d8fdc2513ec6ea8600d76c5e87c45a9a45c16de3e",
    },
    {
        "version": "1.2.0",
        "sha256": "d402a8391ca2624673f124ff42ba8d0d40d4139e5d23111f3995dc6c5f70f63d",
    },
    {
        "version": "1.1.0",
        "sha256": "46d82c9249896903ee6be2295fc52a1346a9ee82f61f89b8a2181232c3bd999b",
    },
    {
        "version": "1.0.0",
        "sha256": "61ef65c738a8cd65059f58f2ee5f7eef493136ac4d5e5c3464787d17043febdf",
    },
    {
        "version": "0.29.1",
        "sha256": "cf0a517f1660a7c4fd26a7ef6f3594bbefcf2b670bc0ed610bf3bb6ec3a9fdc3",
    },
    {
        "version": "0.29.0",
        "sha256": "99d7a6bf9ef0145c59c54b4319fb31cb855681782080a5490909c4a5463c7215",
    },
    {
        "version": "0.28.0",
        "sha256": "64b3fc267fb1f4c56345d96f0ad9f07a2efe43bd15361f818368849cf941b3b7",
    },
    {
        "version": "0.27.0",
        "sha256": "97e2633fefee389daade775da43907aa68699b32212f4e48cb095abe18aa7e65",
    },
    {
        "version": "0.26.0",
        "sha256": "3907dfc6fb27d246e67877e553e8951fac239bb49f2dec7e06b6b09cb0b98b8d",
    },
    {
        "version": "0.25.0",
        "sha256": "e8ab61c047225e808982a564ecd692fd63bd243dccc88a8768ed069a5362a685",
    },
    {
        "version": "0.24.0",
        "sha256": "988fa567906a73e50d3669909285187ef88c76ecd4aa277f4d1f355fc06a90c8",
    },
    {
        "version": "0.23.0",
        "sha256": "56c80fcf49dc606fab8ed5e737a7409e9a486585b7b98673be69b5a4984dd774",
    },
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
