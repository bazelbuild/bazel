// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.common.options;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.InvocationPolicy;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.SetValue.Behavior;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Miscellaneous tests for {@link InvocationPolicy} */
@RunWith(JUnit4.class)
public class InvocationPolicyMiscTest extends InvocationPolicyEnforcerTestBase {

  private static final String BUILD_COMMAND = "build";
  private static final String TEST_DEPRECATED_USER_VALUE = "user value";
  private static final String TEST_DEPRECATED_POLICY_VALUE = "policy value";

  /**
   * Test that deprecated flags set via setValue in the invocation policy don't elicit an extra
   * deprecation warning on top of the one elicted by the user setting the flag.
   */
  @Test
  public void testDoPrintDeprecationWarning_setValue() throws Exception {
    parser.parse("--test_deprecated=" + TEST_DEPRECATED_USER_VALUE);
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("test_deprecated")
        .getUseDefaultBuilder();
    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);

    enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder());

    assertThat(parser.getWarnings())
        .containsExactly(
            "Option 'test_deprecated' is deprecated: Flag for testing deprecation behavior.");
  }

  /**
   * Test that deprecated flags set via UseDefault in the invocation policy don't elicit an extra
   * deprecation warning on top of the one elicted by the user setting the flag.
   */
  @Test
  public void testDoPrintDeprecationWarning_useDefault() throws Exception {
    parser.parse("--test_deprecated=" + TEST_DEPRECATED_USER_VALUE);
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("test_deprecated")
        .getSetValueBuilder()
        .setBehavior(Behavior.FINAL_VALUE_IGNORE_OVERRIDES)
        .addFlagValue(TEST_DEPRECATED_POLICY_VALUE);
    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);

    enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder());

    assertThat(parser.getWarnings())
        .containsExactly(
            "Option 'test_deprecated' is deprecated: Flag for testing deprecation behavior.");
  }

  /**
   * Test that deprecated flags touched via UseDefault in the invocation policy don't elicit a
   * deprecation warning.
   */
  @Test
  public void testDontPrintDeprecationWarning_useDefault() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("test_deprecated")
        .getUseDefaultBuilder();
    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);

    enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder());

    assertThat(parser.getWarnings()).isEmpty();
  }

  /* Test that deprecated flags set via SetValue in the invocation policy don't elicit a
  deprecation warning. */
  @Test
  public void testDontPrintDeprecatioNWarning_setValue() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("test_deprecated")
        .getSetValueBuilder()
        .setBehavior(Behavior.FINAL_VALUE_IGNORE_OVERRIDES)
        .addFlagValue(TEST_DEPRECATED_POLICY_VALUE);
    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);

    enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder());

    assertThat(parser.getWarnings()).isEmpty();
  }
}
