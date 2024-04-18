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
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test InvocationPolicies with the UseDefault operation. */
@RunWith(JUnit4.class)
public class InvocationPolicyUseDefaultTest extends InvocationPolicyEnforcerTestBase {

  // Useful constants
  public static final String BUILD_COMMAND = "build";
  public static final String TEST_STRING_USER_VALUE = "user value";

  @Test
  public void testUseDefault() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder.addFlagPoliciesBuilder()
        .setFlagName("test_string")
        .getUseDefaultBuilder();

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--test_string=" + TEST_STRING_USER_VALUE);

    // Options should be the user specified value before enforcing policy.
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(TEST_STRING_USER_VALUE);

    enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder());

    // Get the options again after policy enforcement: The flag should now be back to its default
    // value
    testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(TestOptions.TEST_STRING_DEFAULT);
  }

  /**
   * Tests UseDefault when the user never actually specified the flag.
   */
  @Test
  public void testUseDefaultWhenFlagWasntSet() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder.addFlagPoliciesBuilder()
        .setFlagName("test_string")
        .getUseDefaultBuilder();

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);

    // Options should be the default since the user never specified it.
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(TestOptions.TEST_STRING_DEFAULT);

    enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder());

    // Still the default.
    testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(TestOptions.TEST_STRING_DEFAULT);
  }

  @Test
  public void testUseDefaultWithExpansionFlags() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder.addFlagPoliciesBuilder()
        .setFlagName("test_expansion")
        .getUseDefaultBuilder();

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--test_expansion");

    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.expandedA).isEqualTo(TestOptions.EXPANDED_A_TEST_EXPANSION);
    assertThat(testOptions.expandedB).isEqualTo(TestOptions.EXPANDED_B_TEST_EXPANSION);
    assertThat(testOptions.expandedC).isEqualTo(TestOptions.EXPANDED_C_TEST_EXPANSION);
    assertThat(testOptions.expandedD).isEqualTo(TestOptions.EXPANDED_D_TEST_EXPANSION);

    enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder());

    // After policy enforcement, all the flags that --test_expansion expanded into should be back
    // to their default values.
    testOptions = getTestOptions();
    assertThat(testOptions.expandedA).isEqualTo(TestOptions.EXPANDED_A_DEFAULT);
    assertThat(testOptions.expandedB).isEqualTo(TestOptions.EXPANDED_B_DEFAULT);
    assertThat(testOptions.expandedC).isEqualTo(TestOptions.EXPANDED_C_DEFAULT);
    assertThat(testOptions.expandedD).isEqualTo(TestOptions.EXPANDED_D_DEFAULT);
  }

  @Test
  public void testUseDefaultWithExpansionFlagAndLaterOverride() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("test_expansion")
        .getUseDefaultBuilder();
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("expanded_b")
        .getAllowValuesBuilder()
        .addAllowedValues("false");

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--test_expansion");

    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.expandedA).isEqualTo(TestOptions.EXPANDED_A_TEST_EXPANSION);
    assertThat(testOptions.expandedB).isEqualTo(TestOptions.EXPANDED_B_TEST_EXPANSION);
    assertThat(testOptions.expandedC).isEqualTo(TestOptions.EXPANDED_C_TEST_EXPANSION);
    assertThat(testOptions.expandedD).isEqualTo(TestOptions.EXPANDED_D_TEST_EXPANSION);

    // If the UseDefault is run, then the value of --expanded_b is back to it's default true, which
    // isn't allowed. However, the allowValues in the later policy should wipe the expansion's
    // policy on --expanded_b, so that the enforcement does not fail.
    enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder());

    testOptions = getTestOptions();
    assertThat(testOptions.expandedA).isEqualTo(TestOptions.EXPANDED_A_DEFAULT);
    assertThat(testOptions.expandedB).isEqualTo(TestOptions.EXPANDED_B_TEST_EXPANSION);
    assertThat(testOptions.expandedC).isEqualTo(TestOptions.EXPANDED_C_DEFAULT);
    assertThat(testOptions.expandedD).isEqualTo(TestOptions.EXPANDED_D_DEFAULT);
  }

  @Test
  public void testUseDefaultWithRecursiveExpansionFlags() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder.addFlagPoliciesBuilder()
        .setFlagName("test_expansion")
        .getUseDefaultBuilder();

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--test_recursive_expansion_top_level");

    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.expandedA).isEqualTo(TestOptions.EXPANDED_A_TEST_RECURSIVE_EXPANSION);
    assertThat(testOptions.expandedB).isEqualTo(TestOptions.EXPANDED_B_TEST_RECURSIVE_EXPANSION);
    assertThat(testOptions.expandedC).isEqualTo(TestOptions.EXPANDED_C_TEST_RECURSIVE_EXPANSION);
    assertThat(testOptions.expandedD).isEqualTo(TestOptions.EXPANDED_D_TEST_RECURSIVE_EXPANSION);

    enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder());

    // After policy enforcement, all the flags that --test_recursive_expansion_top_level and its
    // recursive expansions set should be back to their default values.
    testOptions = getTestOptions();
    assertThat(testOptions.expandedA).isEqualTo(TestOptions.EXPANDED_A_DEFAULT);
    assertThat(testOptions.expandedB).isEqualTo(TestOptions.EXPANDED_B_DEFAULT);
    assertThat(testOptions.expandedC).isEqualTo(TestOptions.EXPANDED_C_DEFAULT);
    assertThat(testOptions.expandedD).isEqualTo(TestOptions.EXPANDED_D_DEFAULT);
  }

  @Test
  public void testUseDefaultWithExpandedFlags() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("expanded_b")
        .getUseDefaultBuilder();

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--test_expansion");

    // --test_expansion should turn set the values from its expansion
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.expandedA).isEqualTo(TestOptions.EXPANDED_A_TEST_EXPANSION);
    assertThat(testOptions.expandedB).isEqualTo(TestOptions.EXPANDED_B_TEST_EXPANSION);
    assertThat(testOptions.expandedC).isEqualTo(TestOptions.EXPANDED_C_TEST_EXPANSION);
    assertThat(testOptions.expandedD).isEqualTo(TestOptions.EXPANDED_D_TEST_EXPANSION);

    enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder());

    // After policy enforcement, expanded_b should be back to its default (true), but the
    // rest should remain the same.
    testOptions = getTestOptions();
    assertThat(testOptions.expandedA).isEqualTo(TestOptions.EXPANDED_A_TEST_EXPANSION);
    assertThat(testOptions.expandedB).isEqualTo(TestOptions.EXPANDED_B_DEFAULT);
    assertThat(testOptions.expandedC).isEqualTo(TestOptions.EXPANDED_C_TEST_EXPANSION);
    assertThat(testOptions.expandedD).isEqualTo(TestOptions.EXPANDED_D_TEST_EXPANSION);
  }

  @Test
  public void testUseDefaultWithFlagWithImplicitRequirements() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder.addFlagPoliciesBuilder()
        .setFlagName("test_implicit_requirement")
        .getUseDefaultBuilder();

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--test_implicit_requirement=" + TEST_STRING_USER_VALUE);

    // test_implicit_requirement sets implicit_requirement_a to "foo", which ignores the user's
    // value because the parser processes implicit values last.
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testImplicitRequirement).isEqualTo(TEST_STRING_USER_VALUE);
    assertThat(testOptions.implicitRequirementA)
        .isEqualTo(TestOptions.IMPLICIT_REQUIREMENT_A_REQUIRED);

    // Then policy puts test_implicit_requirement and its implicit requirements back to its default.
    enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder());

    testOptions = getTestOptions();
    assertThat(testOptions.testImplicitRequirement)
        .isEqualTo(TestOptions.TEST_IMPLICIT_REQUIREMENT_DEFAULT);
    assertThat(testOptions.implicitRequirementA)
        .isEqualTo(TestOptions.IMPLICIT_REQUIREMENT_A_DEFAULT);
  }

  @Test
  public void testUseDefaultWithImplicitlyRequiredFlag() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("implicit_requirement_a")
        .getUseDefaultBuilder();

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse(
        "--test_implicit_requirement=" + TEST_STRING_USER_VALUE,
        "--implicit_requirement_a=" + TEST_STRING_USER_VALUE);

    // test_implicit_requirement sets implicit_requirement_a to "foo", but it gets overwritten
    // by the user value.
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testImplicitRequirement).isEqualTo(TEST_STRING_USER_VALUE);
    assertThat(testOptions.implicitRequirementA).isEqualTo(TEST_STRING_USER_VALUE);

    // Then policy puts implicit_requirement_a back to its default. This is "broken" since it wipes
    // the user value, but this is the behavior that was agreed on and is documented for expansion
    // flags as well.
    enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder());

    testOptions = getTestOptions();
    assertThat(testOptions.testImplicitRequirement).isEqualTo(TEST_STRING_USER_VALUE);
    assertThat(testOptions.implicitRequirementA)
        .isEqualTo(TestOptions.IMPLICIT_REQUIREMENT_A_DEFAULT);
  }

  @Test
  public void testUseDefaultWithFlagWithRecursiveImplicitRequirements() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder.addFlagPoliciesBuilder()
        .setFlagName("test_recursive_implicit_requirement")
        .getUseDefaultBuilder();

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--test_recursive_implicit_requirement=" + TEST_STRING_USER_VALUE);

    // test_recursive_implicit_requirement gets its value from the command line,
    // test_implicit_requirement gets its value from test_recursive_implicit_requirement, and
    // implicit_requirement_a gets its value from test_implicit_requirement.
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testRecursiveImplicitRequirement).isEqualTo(TEST_STRING_USER_VALUE);
    assertThat(testOptions.testImplicitRequirement)
        .isEqualTo(TestOptions.TEST_IMPLICIT_REQUIREMENT_REQUIRED);
    assertThat(testOptions.implicitRequirementA)
        .isEqualTo(TestOptions.IMPLICIT_REQUIREMENT_A_REQUIRED);

    enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder());

    // Policy enforcement should set everything back to its default value.
    testOptions = getTestOptions();
    assertThat(testOptions.testRecursiveImplicitRequirement)
        .isEqualTo(TestOptions.TEST_RECURSIVE_IMPLICIT_REQUIREMENT_DEFAULT);
    assertThat(testOptions.testImplicitRequirement)
        .isEqualTo(TestOptions.TEST_IMPLICIT_REQUIREMENT_DEFAULT);
    assertThat(testOptions.implicitRequirementA)
        .isEqualTo(TestOptions.IMPLICIT_REQUIREMENT_A_DEFAULT);
  }
}
