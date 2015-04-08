#!/bin/bash

# Copyright 2015 Google Inc. All rights reserved.
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

set -eux

# Remove all of the files that we don't want uploaded to GCS. Shuffle the bazel
# binary around so it ends up being the only thing uploaded.
# TODO(kchodorow): change this to clean up everything except bazel-bin when we
# can upload a bootstrapped binary.
mv output/bazel bazel
rm -rf output/* bazel-* src third_party tools examples fromhost scripts site
mv bazel output/bazel
