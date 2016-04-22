// Copyright 2014 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.InvocationPolicy;

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
      "build_interface_so",
      "build-runfiles",
      "namespace-sandbox",
      "process-wrapper",
      "xcode-locator");

  /**
   * Location in the bazel repo where embedded binaries come from.
   */
  public static final ImmutableList<String> EMBEDDED_SCRIPTS_PATHS = ImmutableList.of(
      "src/main/tools");

  /**
   * Path within runfiles tree for finding everything else.
   */
  public static final String RUNFILES_PREFIX = "DOES-NOT-WORK-YET";

  /**
   * Default workspace name.
   */
  public static final String WORKSPACE_NAME = "";

  /**
   * Name of a class with an INSTANCE field of type AnalysisMock to be used for analysis tests.
   */
  public static final String TEST_ANALYSIS_MOCK =
      "com.google.devtools.build.lib.analysis.mock.BazelAnalysisMock";

  /**
   * Directory where we can find bazel's Java tests, relative to a test's runfiles directory.
   */
  public static final String JAVATESTS_ROOT = "src/test/java/";

  public static final String TEST_RULE_CLASS_PROVIDER =
      "com.google.devtools.build.lib.bazel.rules.BazelRuleClassProvider";
  public static final String TEST_RULE_MODULE =
        "com.google.devtools.build.lib.bazel.rules.BazelRulesModule";
  public static final ImmutableList<String> IGNORED_MESSAGE_PREFIXES = ImmutableList.<String>of();

  public static final boolean THIS_IS_BAZEL = true;

  public static final String GCC_INCLUDE_PATH = "external/bazel_tools/tools/cpp/gcc3";

  public static final String TOOLS_REPOSITORY = "@bazel_tools";

  public static final String TOOLS_REPOSITORY_PATH = "tools/cpp";

  public static final ImmutableList<String> DOCS_RULES_PATHS = ImmutableList.of(
      "src/main/java/com/google/devtools/build/lib/rules");

  public static final InvocationPolicy TEST_INVOCATION_POLICY =
      InvocationPolicy.getDefaultInstance();
}
