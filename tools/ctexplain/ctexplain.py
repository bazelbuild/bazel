# Lint as: python3
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
"""ctexplain main entry point.

Currently a stump.
"""
from tools.ctexplain.bazel_api import BazelApi

bazel_api = BazelApi()

# TODO(gregce): move all logic to a _lib library so we can easily include
# end-to-end testing. We'll only handle flag parsing here, which we pass
# into the main invoker as standard Python args.
