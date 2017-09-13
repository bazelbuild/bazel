// Copyright 2016 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.analysis.util;

/**
 * Reference point for expected test failures when trimmed configurations are enabled.
 *
 * <p>Every Bazel test should either succeed with --experimental_dynamic_configs=on or
 * fail with a clear reason due to known features gaps.
 */
public class ExpectedTrimmedConfigurationErrors {
  public static final String LATE_BOUND_ATTRIBUTES_UNSUPPORTED =
      "trimmed configurations don't yet support fragments from late-bound dependencies";

  public static final String MAKE_VARIABLE_FRAGMENTS_UNSUPPORTED =
      "Trimmed configurations don't yet support fragments required by make variables. See "
          + "b/25768144";
}
