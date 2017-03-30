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
package com.google.devtools.build.lib.runtime;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.google.devtools.build.lib.flags.InvocationPolicyEnforcer;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.InvocationPolicy;
import com.google.devtools.common.options.OptionsParsingException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test InvocationPolicies with the SetValues operation. */
@RunWith(JUnit4.class)
public class InvocationPolicySetValueTest extends InvocationPolicyEnforcerTestBase {

  // Useful constants
  public static final String BUILD_COMMAND = "build";
  public static final String TEST_STRING_USER_VALUE = "user value";
  public static final String TEST_STRING_USER_VALUE_2 = "user value 2";
  public static final String TEST_STRING_POLICY_VALUE = "policy value";
  public static final String TEST_STRING_POLICY_VALUE_2 = "policy value 2";

  /**
   * Tests that policy overrides a value when that value is from the user.
   */
  @Test
  public void testSetValueOverridesUser() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("test_string")
        .getSetValueBuilder()
        .addFlagValue(TEST_STRING_POLICY_VALUE);

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--test_string=" + TEST_STRING_USER_VALUE);

    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(TEST_STRING_USER_VALUE);

    enforcer.enforce(parser, BUILD_COMMAND);

    // Get the options again after policy enforcement.
    testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(TEST_STRING_POLICY_VALUE);
  }

  /**
   * Tests that policy overrides a value when the user doesn't specify the value (i.e., the value
   * is from the flag's default from its definition).
   */
  @Test
  public void testSetValueOverridesDefault() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("test_string")
        .getSetValueBuilder()
        .addFlagValue(TEST_STRING_POLICY_VALUE);

    // No user value.
    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);

    // All the flags should be their default value.
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(TestOptions.TEST_STRING_DEFAULT);

    enforcer.enforce(parser, BUILD_COMMAND);

    // Get the options again after policy enforcement.
    testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(TEST_STRING_POLICY_VALUE);
  }

  /**
   * Tests that SetValue overrides the user's value when the flag allows multiple values.
   */
  @Test
  public void testSetValueWithMultipleValuesOverridesUser() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("test_multiple_string")
        .getSetValueBuilder()
        .addFlagValue(TEST_STRING_POLICY_VALUE)
        .addFlagValue(TEST_STRING_POLICY_VALUE_2);

    InvocationPolicyEnforcer enforcer =  createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse(
        "--test_multiple_string=" + TEST_STRING_USER_VALUE,
        "--test_multiple_string=" + TEST_STRING_USER_VALUE_2);

    // Options should not be modified by running the parser through OptionsPolicyEnforcer.create().
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testMultipleString)
        .containsExactly(TEST_STRING_USER_VALUE, TEST_STRING_USER_VALUE_2)
        .inOrder();

    enforcer.enforce(parser, BUILD_COMMAND);

    // Get the options again after policy enforcement.
    testOptions = getTestOptions();
    assertThat(testOptions.testMultipleString)
        .containsExactly(TEST_STRING_POLICY_VALUE, TEST_STRING_POLICY_VALUE_2)
        .inOrder();
  }

  /**
   * Tests that policy overrides the default value when the flag allows multiple values and the user
   * doesn't provide a value.
   */
  @Test
  public void testSetValueWithMultipleValuesOverridesDefault() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("test_multiple_string")
        .getSetValueBuilder()
        .addFlagValue(TEST_STRING_POLICY_VALUE)
        .addFlagValue(TEST_STRING_POLICY_VALUE_2);

    // No user value.
    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);

    // Repeatable flags always default to the empty list.
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testMultipleString).isEmpty();

    enforcer.enforce(parser, BUILD_COMMAND);

    // Options should now be the values from the policy.
    testOptions = getTestOptions();
    assertThat(testOptions.testMultipleString)
        .containsExactly(TEST_STRING_POLICY_VALUE, TEST_STRING_POLICY_VALUE_2)
        .inOrder();
  }

  @Test
  public void testSetValueHasMultipleValuesButFlagIsNotMultiple() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("test_string") // Not repeatable flag.
        .getSetValueBuilder()
        .addFlagValue(TEST_STRING_POLICY_VALUE) // Has multiple values.
        .addFlagValue(TEST_STRING_POLICY_VALUE_2);

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);

    try {
      enforcer.enforce(parser, BUILD_COMMAND);
      fail();
    } catch (OptionsParsingException e) {
      // expected.
    }
  }

  @Test
  public void testSetValueAppendsToMultipleValuedFlag() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("test_multiple_string")
        .getSetValueBuilder()
        .addFlagValue(TEST_STRING_POLICY_VALUE)
        .addFlagValue(TEST_STRING_POLICY_VALUE_2)
        .setAppend(true);

    InvocationPolicyEnforcer enforcer =  createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse(
        "--test_multiple_string=" + TEST_STRING_USER_VALUE,
        "--test_multiple_string=" + TEST_STRING_USER_VALUE_2);

    // Options should not be modified by running the parser through OptionsPolicyEnforcer.create().
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testMultipleString)
        .containsExactly(TEST_STRING_USER_VALUE, TEST_STRING_USER_VALUE_2)
        .inOrder();

    enforcer.enforce(parser, BUILD_COMMAND);

    // Get the options again after policy enforcement.
    testOptions = getTestOptions();
    assertThat(testOptions.testMultipleString)
        .containsExactly(
            TEST_STRING_USER_VALUE,
            TEST_STRING_USER_VALUE_2,
            TEST_STRING_POLICY_VALUE,
            TEST_STRING_POLICY_VALUE_2)
        .inOrder();
  }
  
  @Test
  public void testSetValueWithExpansionFlags() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder.addFlagPoliciesBuilder()
        .setFlagName("test_expansion")
        .getSetValueBuilder()
            .addFlagValue("true"); // this value is arbitrary, the value for a Void flag is ignored

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    // Unrelated flag, but --test_expansion is not set
    parser.parse("--test_string=throwaway value");

    // The flags that --test_expansion expands into should still be their default values
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.expandedA).isEqualTo(TestOptions.EXPANDED_A_DEFAULT);
    assertThat(testOptions.expandedB).isEqualTo(TestOptions.EXPANDED_B_DEFAULT);
    assertThat(testOptions.expandedC).isEqualTo(TestOptions.EXPANDED_C_DEFAULT);
    assertThat(testOptions.expandedD).isEqualTo(TestOptions.EXPANDED_D_DEFAULT);

    enforcer.enforce(parser, BUILD_COMMAND);

    // After policy enforcement, the flags should be the values from --test_expansion
    testOptions = getTestOptions();
    assertThat(testOptions.expandedA).isEqualTo(TestOptions.EXPANDED_A_TEST_EXPANSION);
    assertThat(testOptions.expandedB).isEqualTo(TestOptions.EXPANDED_B_TEST_EXPANSION);
    assertThat(testOptions.expandedC).isEqualTo(TestOptions.EXPANDED_C_TEST_EXPANSION);
    assertThat(testOptions.expandedD).isEqualTo(TestOptions.EXPANDED_D_TEST_EXPANSION);
  }

  @Test
  public void testOverridableSetValueWithExpansionFlags() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("test_expansion")
        .getSetValueBuilder()
        .addFlagValue("") // this value is arbitrary, the value for a Void flag is ignored
        .setOverridable(true);

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    // Unrelated flag, but --test_expansion is not set
    parser.parse("--expanded_c=23");

    // The flags that --test_expansion expands into should still be their default values
    // except for the explicitly marked flag.
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.expandedA).isEqualTo(TestOptions.EXPANDED_B_DEFAULT);
    assertThat(testOptions.expandedB).isEqualTo(TestOptions.EXPANDED_B_DEFAULT);
    assertThat(testOptions.expandedC).isEqualTo(23);
    assertThat(testOptions.expandedD).isEqualTo(TestOptions.EXPANDED_D_DEFAULT);

    enforcer.enforce(parser, "build");

    // After policy enforcement, the flags should be the values from --test_expansion,
    // except for the user-set value, since the expansion flag was set to overridable.
    testOptions = getTestOptions();
    assertThat(testOptions.expandedA).isEqualTo(TestOptions.EXPANDED_A_TEST_EXPANSION);
    assertThat(testOptions.expandedB).isEqualTo(TestOptions.EXPANDED_B_TEST_EXPANSION);
    assertThat(testOptions.expandedC).isEqualTo(23);
    assertThat(testOptions.expandedD).isEqualTo(TestOptions.EXPANDED_D_TEST_EXPANSION);
  }

  @Test
  public void testNonOverridableSetValueWithExpansionFlags() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("test_expansion")
        .getSetValueBuilder()
        .addFlagValue("") // this value is arbitrary, the value for a Void flag is ignored
        .setOverridable(false);

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    // Unrelated flag, but --test_expansion is not set
    parser.parse("--expanded_c=23");

    // The flags that --test_expansion expands into should still be their default values
    // except for the explicitly marked flag.
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.expandedA).isEqualTo(TestOptions.EXPANDED_B_DEFAULT);
    assertThat(testOptions.expandedB).isEqualTo(TestOptions.EXPANDED_B_DEFAULT);
    assertThat(testOptions.expandedC).isEqualTo(23);
    assertThat(testOptions.expandedD).isEqualTo(TestOptions.EXPANDED_D_DEFAULT);

    enforcer.enforce(parser, "build");

    // After policy enforcement, the flags should be the values from --test_expansion,
    // including the value that the user tried to set, since the expansion flag was set
    // non-overridably.
    testOptions = getTestOptions();
    assertThat(testOptions.expandedA).isEqualTo(TestOptions.EXPANDED_A_TEST_EXPANSION);
    assertThat(testOptions.expandedB).isEqualTo(TestOptions.EXPANDED_B_TEST_EXPANSION);
    assertThat(testOptions.expandedC).isEqualTo(TestOptions.EXPANDED_C_TEST_EXPANSION);
    assertThat(testOptions.expandedD).isEqualTo(TestOptions.EXPANDED_D_TEST_EXPANSION);
  }

  @Test
  public void testSetValueWithExpandedFlags() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("expanded_c")
        .getSetValueBuilder()
        .addFlagValue("64");

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--test_expansion");

    // --test_expansion should set the values from its expansion
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.expandedA).isEqualTo(TestOptions.EXPANDED_A_TEST_EXPANSION);
    assertThat(testOptions.expandedB).isEqualTo(TestOptions.EXPANDED_B_TEST_EXPANSION);
    assertThat(testOptions.expandedC).isEqualTo(TestOptions.EXPANDED_C_TEST_EXPANSION);
    assertThat(testOptions.expandedD).isEqualTo(TestOptions.EXPANDED_D_TEST_EXPANSION);

    enforcer.enforce(parser, BUILD_COMMAND);

    // After policy enforcement, expanded_c should be set to 64 from the policy, but the
    // flags should remain the same from the expansion of --test_expansion.
    testOptions = getTestOptions();
    assertThat(testOptions.expandedA).isEqualTo(TestOptions.EXPANDED_A_TEST_EXPANSION);
    assertThat(testOptions.expandedB).isEqualTo(TestOptions.EXPANDED_B_TEST_EXPANSION);
    assertThat(testOptions.expandedC).isEqualTo(64);
    assertThat(testOptions.expandedD).isEqualTo(TestOptions.EXPANDED_D_TEST_EXPANSION);
  }

  @Test
  public void testSetValueWithImplicitlyRequiredFlags() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("implicit_requirement_a")
        .getSetValueBuilder()
        .addFlagValue(TEST_STRING_POLICY_VALUE);

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--test_implicit_requirement=" + TEST_STRING_USER_VALUE);

    // test_implicit_requirement sets implicit_requirement_a to "foo"
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testImplicitRequirement).isEqualTo(TEST_STRING_USER_VALUE);
    assertThat(testOptions.implicitRequirementA)
        .isEqualTo(TestOptions.IMPLICIT_REQUIREMENT_A_REQUIRED);

    enforcer.enforce(parser, BUILD_COMMAND);

    testOptions = getTestOptions();
    assertThat(testOptions.testImplicitRequirement).isEqualTo(TEST_STRING_USER_VALUE);
    assertThat(testOptions.implicitRequirementA).isEqualTo(TEST_STRING_POLICY_VALUE);
  }

  @Test
  public void testSetValueOverridable() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("test_string")
        .getSetValueBuilder()
        .addFlagValue(TEST_STRING_POLICY_VALUE)
        .setOverridable(true);

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--test_string=" + TEST_STRING_USER_VALUE);

    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(TEST_STRING_USER_VALUE);

    enforcer.enforce(parser, BUILD_COMMAND);

    // Even though the policy sets the value for test_string, the policy is overridable and the
    // user set the value, so it should be the user's value.
    testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(TEST_STRING_USER_VALUE);
  }

  @Test
  public void testSetValueWithNoValueThrows() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder.addFlagPoliciesBuilder()
        .setFlagName("test_string")
        .getSetValueBuilder(); // No value.

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--test_string=" + TEST_STRING_USER_VALUE);

    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(TEST_STRING_USER_VALUE);

    try {
      enforcer.enforce(parser, BUILD_COMMAND);
      fail();
    } catch (OptionsParsingException e) {
      // expected.
    }
  }
}
