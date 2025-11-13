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
import static com.google.common.truth.TruthJUnit.assume;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.InvocationPolicy;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.SetValue;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.SetValue.Behavior;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Test InvocationPolicies with the SetValues operation. */
@RunWith(TestParameterInjector.class)
public class InvocationPolicySetValueTest extends InvocationPolicyEnforcerTestBase {

  public static final String BUILD_COMMAND = "build";
  public static final String TEST_STRING_USER_VALUE = "user value";
  public static final String TEST_STRING_USER_VALUE_2 = "user value 2";
  public static final String TEST_STRING_POLICY_VALUE = "policy value";
  public static final String TEST_STRING_POLICY_VALUE_2 = "policy value 2";

  /** Tests that policy overwrites a value when that value is from the user. */
  @Test
  public void finalValueIgnoreOverrides_overwritesUserSetting() throws Exception {
    InvocationPolicy.Builder invocationPolicy = InvocationPolicy.newBuilder();
    invocationPolicy
        .addFlagPoliciesBuilder()
        .setFlagName("test_string")
        .getSetValueBuilder()
        .setBehavior(Behavior.FINAL_VALUE_IGNORE_OVERRIDES)
        .addFlagValue(TEST_STRING_POLICY_VALUE);

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicy);
    parser.parse("--test_string=" + TEST_STRING_USER_VALUE);

    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(TEST_STRING_USER_VALUE);

    enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder());

    // Get the options again after policy enforcement.
    testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(TEST_STRING_POLICY_VALUE);
    assertThat(
            parser.asCompleteListOfParsedOptions().stream()
                .map(ParsedOptionDescription::getCommandLineForm))
        .containsExactly(
            "--test_string=" + TEST_STRING_USER_VALUE, "--test_string=" + TEST_STRING_POLICY_VALUE)
        .inOrder();
  }

  /**
   * Tests that policy overwrites a value when the user doesn't specify the value (i.e., the value
   * is from the flag's default from its definition).
   */
  @Test
  public void finalValueIgnoreOverrides_overwritesDefault() throws Exception {
    InvocationPolicy.Builder invocationPolicy = InvocationPolicy.newBuilder();
    invocationPolicy
        .addFlagPoliciesBuilder()
        .setFlagName("test_string")
        .getSetValueBuilder()
        .setBehavior(Behavior.FINAL_VALUE_IGNORE_OVERRIDES)
        .addFlagValue(TEST_STRING_POLICY_VALUE);

    // No user value.
    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicy);

    // All the flags should be their default value.
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(TestOptions.TEST_STRING_DEFAULT);

    enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder());

    // Get the options again after policy enforcement.
    testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(TEST_STRING_POLICY_VALUE);
  }

  /** Tests that SetValue overwrites the user's value when the flag allows multiple values. */
  @Test
  public void finalValueIgnoreOverrides_flagWithMultipleValues_overwritesUserSetting()
      throws Exception {
    InvocationPolicy.Builder invocationPolicy = InvocationPolicy.newBuilder();
    invocationPolicy
        .addFlagPoliciesBuilder()
        .setFlagName("test_multiple_string")
        .getSetValueBuilder()
        .setBehavior(Behavior.FINAL_VALUE_IGNORE_OVERRIDES)
        .addFlagValue(TEST_STRING_POLICY_VALUE)
        .addFlagValue(TEST_STRING_POLICY_VALUE_2);

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicy);
    parser.parse(
        "--test_multiple_string=" + TEST_STRING_USER_VALUE,
        "--test_multiple_string=" + TEST_STRING_USER_VALUE_2);

    // Options should not be modified by running the parser through OptionsPolicyEnforcer.create().
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testMultipleString)
        .containsExactly(TEST_STRING_USER_VALUE, TEST_STRING_USER_VALUE_2)
        .inOrder();

    enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder());

    // Get the options again after policy enforcement.
    testOptions = getTestOptions();
    assertThat(testOptions.testMultipleString)
        .containsExactly(TEST_STRING_POLICY_VALUE, TEST_STRING_POLICY_VALUE_2)
        .inOrder();
  }

  /**
   * Tests that policy overwrites the default value when the flag allows multiple values and the
   * user doesn't provide a value.
   */
  @Test
  public void finalValueIgnoreOverrides_flagWithMultipleValues_overwritesDefault()
      throws Exception {
    InvocationPolicy.Builder invocationPolicy = InvocationPolicy.newBuilder();
    invocationPolicy
        .addFlagPoliciesBuilder()
        .setFlagName("test_multiple_string")
        .getSetValueBuilder()
        .setBehavior(Behavior.FINAL_VALUE_IGNORE_OVERRIDES)
        .addFlagValue(TEST_STRING_POLICY_VALUE)
        .addFlagValue(TEST_STRING_POLICY_VALUE_2);

    // No user value.
    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicy);

    // Repeatable flags always default to the empty list.
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testMultipleString).isEmpty();

    enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder());

    // Options should now be the values from the policy.
    testOptions = getTestOptions();
    assertThat(testOptions.testMultipleString)
        .containsExactly(TEST_STRING_POLICY_VALUE, TEST_STRING_POLICY_VALUE_2)
        .inOrder();
  }

  @Test
  public void setMultipleValuesForSingleValuedFlag_fails(@TestParameter Behavior behavior)
      throws Exception {
    assume().that(behavior).isNotEqualTo(Behavior.UNDEFINED);
    InvocationPolicy.Builder invocationPolicy = InvocationPolicy.newBuilder();
    invocationPolicy
        .addFlagPoliciesBuilder()
        .setFlagName("test_string") // Not repeatable flag.
        .getSetValueBuilder()
        .setBehavior(behavior)
        .addFlagValue(TEST_STRING_POLICY_VALUE) // Has multiple values.
        .addFlagValue(TEST_STRING_POLICY_VALUE_2);

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicy);

    assertThrows(
        OptionsParsingException.class,
        () -> enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder()));
  }

  @Test
  public void append_appendsToMultipleValuedFlag() throws Exception {
    InvocationPolicy.Builder invocationPolicy = InvocationPolicy.newBuilder();
    invocationPolicy
        .addFlagPoliciesBuilder()
        .setFlagName("test_multiple_string")
        .getSetValueBuilder()
        .setBehavior(Behavior.APPEND)
        .addFlagValue(TEST_STRING_POLICY_VALUE)
        .addFlagValue(TEST_STRING_POLICY_VALUE_2);

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicy);
    parser.parse(
        "--test_multiple_string=" + TEST_STRING_USER_VALUE,
        "--test_multiple_string=" + TEST_STRING_USER_VALUE_2);

    // Options should not be modified by running the parser through OptionsPolicyEnforcer.create().
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testMultipleString)
        .containsExactly(TEST_STRING_USER_VALUE, TEST_STRING_USER_VALUE_2)
        .inOrder();

    enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder());

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
  public void setFlagWithExpansion_finalValueIgnoreOverrides_setsExpandedValuesAsFinal(
      @TestParameter({"null", "", "some value"}) String value) throws Exception {
    InvocationPolicy.Builder invocationPolicy = InvocationPolicy.newBuilder();
    SetValue.Builder setValue =
        invocationPolicy
            .addFlagPoliciesBuilder()
            .setFlagName("test_expansion")
            // SetValue must have no values for a Void flag.
            .getSetValueBuilder()
            .setBehavior(Behavior.FINAL_VALUE_IGNORE_OVERRIDES);
    if (value != null) {
      setValue.addFlagValue(value);
    }

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicy);
    // Unrelated flag, but --test_expansion is not set
    parser.parse("--test_string=throwaway value");

    // The flags that --test_expansion expands into should still be their default values
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.expandedA).isEqualTo(TestOptions.EXPANDED_A_DEFAULT);
    assertThat(testOptions.expandedB).isEqualTo(TestOptions.EXPANDED_B_DEFAULT);
    assertThat(testOptions.expandedC).isEqualTo(TestOptions.EXPANDED_C_DEFAULT);
    assertThat(testOptions.expandedD).isEqualTo(TestOptions.EXPANDED_D_DEFAULT);

    enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder());

    // After policy enforcement, the flags should be the values from --test_expansion
    testOptions = getTestOptions();
    assertThat(testOptions.expandedA).isEqualTo(TestOptions.EXPANDED_A_TEST_EXPANSION);
    assertThat(testOptions.expandedB).isEqualTo(TestOptions.EXPANDED_B_TEST_EXPANSION);
    assertThat(testOptions.expandedC).isEqualTo(TestOptions.EXPANDED_C_TEST_EXPANSION);
    assertThat(testOptions.expandedD).isEqualTo(TestOptions.EXPANDED_D_TEST_EXPANSION);
  }

  @Test
  public void finalValueIgnoreOverrides_flagExpandingToExpansion_setsRecursiveValues()
      throws Exception {
    InvocationPolicy.Builder invocationPolicy = InvocationPolicy.newBuilder();
    invocationPolicy
        .addFlagPoliciesBuilder()
        .setFlagName("test_recursive_expansion_top_level")
        .getSetValueBuilder()
        .setBehavior(Behavior.FINAL_VALUE_IGNORE_OVERRIDES);

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicy);
    // Unrelated flag, but --test_expansion is not set
    parser.parse("--test_string=throwaway value");

    // The flags that --test_expansion expands into should still be their default values
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.expandedA).isEqualTo(TestOptions.EXPANDED_A_DEFAULT);
    assertThat(testOptions.expandedB).isEqualTo(TestOptions.EXPANDED_B_DEFAULT);
    assertThat(testOptions.expandedC).isEqualTo(TestOptions.EXPANDED_C_DEFAULT);
    assertThat(testOptions.expandedD).isEqualTo(TestOptions.EXPANDED_D_DEFAULT);

    enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder());

    // After policy enforcement, the flags should be the values from the expansion flag
    testOptions = getTestOptions();
    assertThat(testOptions.expandedA).isEqualTo(TestOptions.EXPANDED_A_TEST_RECURSIVE_EXPANSION);
    assertThat(testOptions.expandedB).isEqualTo(TestOptions.EXPANDED_B_TEST_RECURSIVE_EXPANSION);
    assertThat(testOptions.expandedC).isEqualTo(TestOptions.EXPANDED_C_TEST_RECURSIVE_EXPANSION);
    assertThat(testOptions.expandedD).isEqualTo(TestOptions.EXPANDED_D_TEST_RECURSIVE_EXPANSION);
  }

  @Test
  public void allowOverrides_setFlagWithExpansion_keepsUserSpecifiedFlag() throws Exception {
    InvocationPolicy.Builder invocationPolicy = InvocationPolicy.newBuilder();
    invocationPolicy
        .addFlagPoliciesBuilder()
        .setFlagName("test_expansion")
        .getSetValueBuilder()
        .setBehavior(Behavior.ALLOW_OVERRIDES)
        .addFlagValue(""); // this value is arbitrary, the value for a Void flag is ignored

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicy);
    // Unrelated flag, but --test_expansion is not set
    parser.parse("--expanded_c=23");

    // The flags that --test_expansion expands into should still be their default values
    // except for the explicitly marked flag.
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.expandedA).isEqualTo(TestOptions.EXPANDED_B_DEFAULT);
    assertThat(testOptions.expandedB).isEqualTo(TestOptions.EXPANDED_B_DEFAULT);
    assertThat(testOptions.expandedC).isEqualTo(23);
    assertThat(testOptions.expandedD).isEqualTo(TestOptions.EXPANDED_D_DEFAULT);

    enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder());

    // After policy enforcement, the flags should be the values from --test_expansion,
    // except for the user-set value, since the expansion flag was set to overridable.
    testOptions = getTestOptions();
    assertThat(testOptions.expandedA).isEqualTo(TestOptions.EXPANDED_A_TEST_EXPANSION);
    assertThat(testOptions.expandedB).isEqualTo(TestOptions.EXPANDED_B_TEST_EXPANSION);
    assertThat(testOptions.expandedC).isEqualTo(23);
    assertThat(testOptions.expandedD).isEqualTo(TestOptions.EXPANDED_D_TEST_EXPANSION);
  }

  @Test
  public void allowOverrides_flagExpandingToRepeatingFlag_appendsRepeatedValues() throws Exception {
    InvocationPolicy.Builder invocationPolicy = InvocationPolicy.newBuilder();
    invocationPolicy
        .addFlagPoliciesBuilder()
        .setFlagName("test_expansion_to_repeatable")
        .getSetValueBuilder()
        // SetValue must have no values for a Void flag.
        .setBehavior(Behavior.ALLOW_OVERRIDES);

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicy);
    // Unrelated flag, but --test_expansion is not set
    parser.parse("--test_multiple_string=foo");

    // The flags that --test_expansion expands into should still be their default values
    // except for the explicitly marked flag.
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testMultipleString).containsExactly("foo");

    enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder());

    // After policy enforcement, the flags should be the values from --test_expansion,
    // except for the user-set value, since the expansion flag was set to overridable.
    testOptions = getTestOptions();
    assertThat(testOptions.testMultipleString).containsExactly(
        "foo", TestOptions.EXPANDED_MULTIPLE_1, TestOptions.EXPANDED_MULTIPLE_2);
  }

  @Test
  public void finalValueIgnoreOverrides_setFlagWithExpansion_overwritesUserSettingForExpandedFlag()
      throws Exception {
    InvocationPolicy.Builder invocationPolicy = InvocationPolicy.newBuilder();
    invocationPolicy
        .addFlagPoliciesBuilder()
        .setFlagName("test_expansion")
        // SetValue must have no values for a Void flag.
        .getSetValueBuilder()
        .setBehavior(Behavior.FINAL_VALUE_IGNORE_OVERRIDES);

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicy);
    // Unrelated flag, but --test_expansion is not set
    parser.parse("--expanded_c=23");

    // The flags that --test_expansion expands into should still be their default values
    // except for the explicitly marked flag.
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.expandedA).isEqualTo(TestOptions.EXPANDED_B_DEFAULT);
    assertThat(testOptions.expandedB).isEqualTo(TestOptions.EXPANDED_B_DEFAULT);
    assertThat(testOptions.expandedC).isEqualTo(23);
    assertThat(testOptions.expandedD).isEqualTo(TestOptions.EXPANDED_D_DEFAULT);

    enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder());

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
  public void finalValueIgnoreOverrides_flagWithExpansionToRepeatingFlag_overwritesUserSetting()
      throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("test_expansion_to_repeatable")
        // SetValue must have no values for a Void flag.
        .getSetValueBuilder()
        .setBehavior(Behavior.FINAL_VALUE_IGNORE_OVERRIDES);

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    // Unrelated flag, but --test_expansion is not set
    parser.parse("--test_multiple_string=foo");

    // The flags that --test_expansion expands into should still be their default values
    // except for the explicitly marked flag.
    assertThat(getTestOptions().testMultipleString).contains("foo");

    enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder());

    // After policy enforcement, the flag should no longer have the user's value.
    assertThat(getTestOptions().testMultipleString)
        .containsExactly(TestOptions.EXPANDED_MULTIPLE_1, TestOptions.EXPANDED_MULTIPLE_2);
  }

  @Test
  public void finalValueIgnoreOverrides_overwritesUserSettingFromExpandedFlag() throws Exception {
    InvocationPolicy.Builder invocationPolicy = InvocationPolicy.newBuilder();
    invocationPolicy
        .addFlagPoliciesBuilder()
        .setFlagName("expanded_c")
        .getSetValueBuilder()
        .setBehavior(Behavior.FINAL_VALUE_IGNORE_OVERRIDES)
        .addFlagValue("64");

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicy);
    parser.parse("--test_expansion");

    // --test_expansion should set the values from its expansion
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.expandedA).isEqualTo(TestOptions.EXPANDED_A_TEST_EXPANSION);
    assertThat(testOptions.expandedB).isEqualTo(TestOptions.EXPANDED_B_TEST_EXPANSION);
    assertThat(testOptions.expandedC).isEqualTo(TestOptions.EXPANDED_C_TEST_EXPANSION);
    assertThat(testOptions.expandedD).isEqualTo(TestOptions.EXPANDED_D_TEST_EXPANSION);

    enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder());

    // After policy enforcement, expanded_c should be set to 64 from the policy, but the
    // flags should remain the same from the expansion of --test_expansion.
    testOptions = getTestOptions();
    assertThat(testOptions.expandedA).isEqualTo(TestOptions.EXPANDED_A_TEST_EXPANSION);
    assertThat(testOptions.expandedB).isEqualTo(TestOptions.EXPANDED_B_TEST_EXPANSION);
    assertThat(testOptions.expandedC).isEqualTo(64);
    assertThat(testOptions.expandedD).isEqualTo(TestOptions.EXPANDED_D_TEST_EXPANSION);
  }

  @Test
  public void finalValueIgnoreOverrides_overwritesImplicitRequirementFromUserSetting()
      throws Exception {
    InvocationPolicy.Builder invocationPolicy = InvocationPolicy.newBuilder();
    invocationPolicy
        .addFlagPoliciesBuilder()
        .setFlagName("implicit_requirement_a")
        .getSetValueBuilder()
        .setBehavior(Behavior.FINAL_VALUE_IGNORE_OVERRIDES)
        .addFlagValue(TEST_STRING_POLICY_VALUE);

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicy);
    parser.parse("--test_implicit_requirement=" + TEST_STRING_USER_VALUE);

    // test_implicit_requirement sets implicit_requirement_a to "foo"
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testImplicitRequirement).isEqualTo(TEST_STRING_USER_VALUE);
    assertThat(testOptions.implicitRequirementA)
        .isEqualTo(TestOptions.IMPLICIT_REQUIREMENT_A_REQUIRED);

    enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder());

    testOptions = getTestOptions();
    assertThat(testOptions.testImplicitRequirement).isEqualTo(TEST_STRING_USER_VALUE);
    assertThat(testOptions.implicitRequirementA).isEqualTo(TEST_STRING_POLICY_VALUE);
  }

  @Test
  public void testSetValueWithImplicitlyRequiredFlags() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("test_implicit_requirement")
        .getSetValueBuilder()
        .setBehavior(Behavior.FINAL_VALUE_IGNORE_OVERRIDES)
        .addFlagValue(TEST_STRING_POLICY_VALUE);
    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--implicit_requirement_a=" + TEST_STRING_USER_VALUE);
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.implicitRequirementA).isEqualTo(TEST_STRING_USER_VALUE);

    enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder());

    testOptions = getTestOptions();
    assertThat(testOptions.testImplicitRequirement).isEqualTo(TEST_STRING_POLICY_VALUE);
    assertThat(testOptions.implicitRequirementA)
        .isEqualTo(TestOptions.IMPLICIT_REQUIREMENT_A_REQUIRED);
  }

  @Test
  public void allowOverrides_leavesUserSetting() throws Exception {
    InvocationPolicy.Builder invocationPolicy = InvocationPolicy.newBuilder();
    invocationPolicy
        .addFlagPoliciesBuilder()
        .setFlagName("test_string")
        .getSetValueBuilder()
        .setBehavior(Behavior.ALLOW_OVERRIDES)
        .addFlagValue(TEST_STRING_POLICY_VALUE);

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicy);
    parser.parse("--test_string=" + TEST_STRING_USER_VALUE);

    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(TEST_STRING_USER_VALUE);

    enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder());

    // Even though the policy sets the value for test_string, the policy is overridable and the
    // user set the value, so it should be the user's value.
    testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(TEST_STRING_USER_VALUE);
  }

  @Test
  public void setFlagValueWithNoValue_fails(@TestParameter Behavior behavior) throws Exception {
    assume().that(behavior).isNotEqualTo(Behavior.UNDEFINED);
    InvocationPolicy.Builder invocationPolicy = InvocationPolicy.newBuilder();
    invocationPolicy
        .addFlagPoliciesBuilder()
        .setFlagName("test_string")
        .getSetValueBuilder()
        .setBehavior(behavior); // No value.

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicy);
    parser.parse("--test_string=" + TEST_STRING_USER_VALUE);

    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(TEST_STRING_USER_VALUE);

    assertThrows(
        OptionsParsingException.class,
        () -> enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder()));
  }

  @Test
  public void enforce_setValueWithUndefinedBehavior_fails(
      @TestParameter boolean hasBehavior,
      @TestParameter({"test_string", "test_expansion", "test_implicit_requirement"})
          String flagName)
      throws Exception {
    InvocationPolicy.Builder invocationPolicy = InvocationPolicy.newBuilder();
    SetValue.Builder setValue =
        invocationPolicy
            .addFlagPoliciesBuilder()
            .setFlagName(flagName)
            .getSetValueBuilder()
            .addFlagValue("any value");
    if (hasBehavior) {
      setValue.setBehavior(Behavior.UNDEFINED);
    }
    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicy);

    OptionsParsingException e =
        assertThrows(
            OptionsParsingException.class,
            () -> enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder()));
    assertThat(e)
        .hasMessageThat()
        .startsWith(
            String.format(
                "SetValue operation from invocation policy for has an undefined behavior:"
                    + " flag_name: \"%s\"\n"
                    + "set_value {\n",
                flagName));
  }

  @Test
  public void enforce_policySettingConfig_fails(@TestParameter Behavior behavior) throws Exception {
    assume().that(behavior).isNotEqualTo(Behavior.UNDEFINED);
    InvocationPolicy.Builder invocationPolicy = InvocationPolicy.newBuilder();
    invocationPolicy
        .addFlagPoliciesBuilder()
        .setFlagName("config")
        .getSetValueBuilder()
        .setBehavior(behavior)
        .addFlagValue("foo");

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicy);
    parser.parse();
    OptionsParsingException expected =
        assertThrows(
            OptionsParsingException.class,
            () -> enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder()));
    assertThat(expected)
        .hasMessageThat()
        .startsWith(
            "Invocation policy is applied after --config expansion, changing config values now "
                + "would have no effect and is disallowed to prevent confusion. Please remove "
                + "the following policy : flag_name: \"config\"\n"
                + "set_value {\n"
                + "  flag_value: \"foo\"\n"
                + "  behavior: "
                + behavior
                + "\n");
  }

  @Test
  public void enforce_setValueForNonexistentFlag_doesNothing(@TestParameter Behavior behavior)
      throws Exception {
    assume().that(behavior).isNotEqualTo(Behavior.UNDEFINED);
    InvocationPolicy.Builder invocationPolicy = InvocationPolicy.newBuilder();
    invocationPolicy
        .addFlagPoliciesBuilder()
        .setFlagName("nonexistent")
        .getSetValueBuilder()
        .setBehavior(behavior)
        .addFlagValue("hello");
    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicy);
    parser.parse();

    enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder());

    assertThat(getTestOptions()).isEqualTo(Options.getDefaults(TestOptions.class));
  }

  @Test
  public void finalValueThrowOnOverride_throwsOnUserOverride() throws Exception {
    InvocationPolicy.Builder invocationPolicy = InvocationPolicy.newBuilder();
    invocationPolicy
        .addFlagPoliciesBuilder()
        .setFlagName("test_string")
        .setCustomErrorMessage("See {link to test_string policy} for more details.")
        .getSetValueBuilder()
        .setBehavior(Behavior.FINAL_VALUE_THROW_ON_OVERRIDE)
        .addFlagValue(TEST_STRING_POLICY_VALUE);
    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicy);
    parser.parse("--test_string=" + TEST_STRING_USER_VALUE);

    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(TEST_STRING_USER_VALUE);

    OptionsParsingException e =
        assertThrows(
            OptionsParsingException.class,
            () -> enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder()));

    assertThat(e)
        .hasMessageThat()
        .isEqualTo(
            "User set a value for option '--test_string' which is not permitted by the invocation"
                + " policy. This flag value will always be overridden to [policy value]. See"
                + " {link to test_string policy} for more details.");
  }

  @Test
  public void finalValueThrowOnOverride_successOnNoUserOverride() throws Exception {
    InvocationPolicy.Builder invocationPolicy = InvocationPolicy.newBuilder();
    invocationPolicy
        .addFlagPoliciesBuilder()
        .setFlagName("test_string")
        .setCustomErrorMessage("See {link to test_string policy} for more details.")
        .getSetValueBuilder()
        .setBehavior(Behavior.FINAL_VALUE_THROW_ON_OVERRIDE)
        .addFlagValue(TEST_STRING_POLICY_VALUE);
    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicy);

    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo("test string default");

    enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder());

    // Get the options again after policy enforcement.
    testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(TEST_STRING_POLICY_VALUE);
    assertThat(
            parser.asCompleteListOfParsedOptions().stream()
                .map(ParsedOptionDescription::getCommandLineForm))
        .containsExactly("--test_string=" + TEST_STRING_POLICY_VALUE)
        .inOrder();
  }

  @Test
  public void finalValueThrowOnOverride_flagWithMultipleValues_throwsOnUserOverride()
      throws Exception {
    InvocationPolicy.Builder invocationPolicy = InvocationPolicy.newBuilder();
    invocationPolicy
        .addFlagPoliciesBuilder()
        .setFlagName("test_multiple_string")
        .setCustomErrorMessage("See {link to test_multiple_string policy} for more details.")
        .getSetValueBuilder()
        .setBehavior(Behavior.FINAL_VALUE_THROW_ON_OVERRIDE)
        .addFlagValue(TEST_STRING_POLICY_VALUE)
        .addFlagValue(TEST_STRING_POLICY_VALUE_2);

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicy);
    parser.parse(
        "--test_multiple_string=" + TEST_STRING_USER_VALUE,
        "--test_multiple_string=" + TEST_STRING_USER_VALUE_2);

    // Options should not be modified by running the parser through OptionsPolicyEnforcer.create().
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testMultipleString)
        .containsExactly(TEST_STRING_USER_VALUE, TEST_STRING_USER_VALUE_2)
        .inOrder();

    OptionsParsingException e =
        assertThrows(
            OptionsParsingException.class,
            () -> enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder()));

    assertThat(e)
        .hasMessageThat()
        .isEqualTo(
            "User set a value for option '--test_multiple_string' which is not permitted by the"
                + " invocation policy. This flag value will always be overridden to [policy value,"
                + " policy value 2]. See {link to test_multiple_string policy} for more details.");
  }

  @Test
  public void finalValueThrowOnOverride_flagWithMultipleValues_successOnNoUserOverride()
      throws Exception {
    InvocationPolicy.Builder invocationPolicy = InvocationPolicy.newBuilder();
    invocationPolicy
        .addFlagPoliciesBuilder()
        .setFlagName("test_multiple_string")
        .getSetValueBuilder()
        .setBehavior(Behavior.FINAL_VALUE_THROW_ON_OVERRIDE)
        .addFlagValue(TEST_STRING_POLICY_VALUE)
        .addFlagValue(TEST_STRING_POLICY_VALUE_2);

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicy);

    // Options should not be modified by running the parser through OptionsPolicyEnforcer.create().
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testMultipleString).isEmpty();

    enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder());

    // Get the options again after policy enforcement.
    testOptions = getTestOptions();
    assertThat(testOptions.testMultipleString)
        .containsExactly(TEST_STRING_POLICY_VALUE, TEST_STRING_POLICY_VALUE_2)
        .inOrder();
  }
}
