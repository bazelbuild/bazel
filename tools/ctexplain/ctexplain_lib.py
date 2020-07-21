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
"""ctexplain business logic."""
from typing import Tuple

from tools.ctexplain.bazel_api import BazelApi
from tools.ctexplain.types import ConfiguredTarget


def analyze_build(bazel_api: BazelApi, labels: Tuple[str, ...],
                  build_flags: Tuple[str, ...]) -> Tuple[ConfiguredTarget, ...]:
    """Documentation here"""
    cquery_args = [f'deps({",".join(labels)})']
    cquery_args.extend(build_flags)
    (success, stderr, cts) = bazel_api.cquery(cquery_args)
    print(f"success: {success}")
    print(f"stderr: {stderr}")
    print(f"CTs: {len(cts)}")
    return cts


# Get a build's CTs
# Report basic statistics
# Get the trimmed CT graph
# Report comparative statistics

# Collect results into a machine-readable data structure
# Pass through user outputters for user-friendly output.

# TODO(gregce): move all logic to a _lib library so we can easily include
# end-to-end testing. We'll only handle flag parsing here, which we pass
# into the main invoker as standard Python args.

