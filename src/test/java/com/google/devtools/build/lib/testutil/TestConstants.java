// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.testutil;

import com.google.common.collect.ImmutableList;

/**
 * Various constants required by the tests.
 */
public class TestConstants {
  private TestConstants() {
  }

  /**
   * A list of all embedded binaries that go into the regular Bazel binary.
   */
  public static final ImmutableList<String> EMBEDDED_TOOLS = ImmutableList.of(
      "build-runfiles",
      "process-wrapper",
      "build_interface_so");

  /**
   * Location in the bazel repo where embedded binaries come from.
   */
  public static final ImmutableList<String> EMBEDDED_SCRIPTS_PATHS = ImmutableList.of(
      "DOES-NOT-WORK-YET");

  /**
   * Path within runfiles tree for finding everything else.
   */
  public static final String RUNFILES_PREFIX = "DOES-NOT-WORK-YET";

  /**
   * Name of a class with an INSTANCE field of type AnalysisMock to be used for analysis tests.
   */
  public static final String TEST_ANALYSIS_MOCK =
      "com.google.devtools.build.lib.analysis.BazelAnalysisMock";

  /**
   * Directory where we can find bazel's Java tests, relative to a test's runfiles directory.
   */
  public static final String JAVATESTS_ROOT = "src/test/java/";

  public static final String TEST_RULE_CLASS_PROVIDER =
      "com.google.devtools.build.lib.bazel.rules.BazelRuleClassProvider";
  public static final ImmutableList<String> IGNORED_MESSAGE_PREFIXES = ImmutableList.<String>of();
}
