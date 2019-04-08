#!/bin/bash
#
# Copyright 2019 The Bazel Authors. All rights reserved.
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

gsutil cp "tools/test/CoverageOutputGenerator/java/com/google/devtools/coverageoutputgenerator/coverage_output_generator.zip" \
  gs://bazel-mirror/bazel_coverage_output_generator/coverage_output_generator-$(git rev-parse HEAD)-$(date +%s).zip
