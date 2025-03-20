// Copyright 2017 The Bazel Authors. All rights reserved.
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

/** Test InvocationPolicies on cases where we expect it to fail gracefully. */
@RunWith(JUnit4.class)
public class InvocationPolicyBreakingConditionsTest extends InvocationPolicyEnforcerTestBase {

  // Useful constants
  public static final String TEST_STRING_USER_VALUE = "user value";
  public static final String TEST_STRING_POLICY_VALUE = "policy value";
  public static final String TEST_STRING_POLICY_VALUE_2 = "policy value 2";

  @Test
  public void testFlagPolicyDoesNotApply() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("test_string")
        .addCommands("build")
        .getSetValueBuilder()
        .addFlagValue(TEST_STRING_POLICY_VALUE);

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--test_string=" + TEST_STRING_USER_VALUE);

    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(TEST_STRING_USER_VALUE);

    enforcer.enforce(parser, "test", ImmutableList.builder());

    // Still user value.
    testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(TEST_STRING_USER_VALUE);
  }

  @Test
  public void testNonExistantFlagFromPolicy() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("i_do_not_exist")
        .getSetValueBuilder()
        .setBehavior(Behavior.FINAL_VALUE_IGNORE_OVERRIDES)
        .addFlagValue(TEST_STRING_POLICY_VALUE);
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("test_string")
        .getSetValueBuilder()
        .setBehavior(Behavior.FINAL_VALUE_IGNORE_OVERRIDES)
        .addFlagValue(TEST_STRING_POLICY_VALUE_2);

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--test_string=" + TEST_STRING_USER_VALUE);

    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(TEST_STRING_USER_VALUE);

    enforcer.enforce(parser, "test", ImmutableList.builder());

    // Still user value.
    testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(TEST_STRING_POLICY_VALUE_2);
  }

  @Test
  public void testOperationNotSet() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder.addFlagPoliciesBuilder();
    // No operations added to the flag policy

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--test_string=" + TEST_STRING_USER_VALUE);

    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(TEST_STRING_USER_VALUE);

    // Shouldn't throw.
    enforcer.enforce(parser, "test", ImmutableList.builder());

    // Still user value.
    testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(TEST_STRING_USER_VALUE);
  }
}
