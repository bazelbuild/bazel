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
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.DisallowValues;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.InvocationPolicy;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.UseDefault;
import java.util.Arrays;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test InvocationPolicies with the DisallowValues operation. */
@RunWith(JUnit4.class)
public class InvocationPolicyDisallowValuesTest extends InvocationPolicyEnforcerTestBase {

  // Useful constants
  public static final String BUILD_COMMAND = "build";
  public static final String DISALLOWED_VALUE_1 = "foo";
  public static final String DISALLOWED_VALUE_2 = "bar";
  public static final String UNFILTERED_VALUE = "baz";

  @Test
  public void testDisallowValuesAllowsValue() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("test_string")
        .getDisallowValuesBuilder()
        .addDisallowedValues(DISALLOWED_VALUE_1)
        .addDisallowedValues(DISALLOWED_VALUE_2);

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--test_string=" + UNFILTERED_VALUE);

    // Option should be "baz" as specified by the user.
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(UNFILTERED_VALUE);

    enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder());

    // Still "baz" since "baz" is allowed by the policy.
    testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(UNFILTERED_VALUE);
  }

  @Test
  public void testDisallowValuesDisallowsValue() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("test_string")
        .getDisallowValuesBuilder()
        .addDisallowedValues(DISALLOWED_VALUE_1)
        .addDisallowedValues(DISALLOWED_VALUE_2);

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--test_string=" + DISALLOWED_VALUE_1);

    // Option should be "foo" as specified by the user.
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(DISALLOWED_VALUE_1);

    // expected, since foo is disallowed.
    assertThrows(
        OptionsParsingException.class,
        () -> enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder()));
  }

  @Test
  public void testDisallowValuesDisallowsMultipleValues() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("test_multiple_string")
        .getDisallowValuesBuilder()
        .addDisallowedValues(DISALLOWED_VALUE_1)
        .addDisallowedValues(DISALLOWED_VALUE_2);

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse(
        "--test_multiple_string=" + UNFILTERED_VALUE,
        "--test_multiple_string=" + DISALLOWED_VALUE_2);

    // Option should be "baz" and "bar" as specified by the user.
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testMultipleString)
        .containsExactly(UNFILTERED_VALUE, DISALLOWED_VALUE_2)
        .inOrder();

    // expected, since bar is disallowed.
    assertThrows(
        OptionsParsingException.class,
        () -> enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder()));
  }

  @Test
  public void testDisallowValuesSetsNewValue() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("test_string")
        .getDisallowValuesBuilder()
        .addDisallowedValues(DISALLOWED_VALUE_1)
        .setNewValue(UNFILTERED_VALUE);

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--test_string=" + DISALLOWED_VALUE_1);

    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(DISALLOWED_VALUE_1);

    enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder());

    // Should now be "baz" because the policy forces disallowed values to "baz"
    testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(UNFILTERED_VALUE);
  }

  @Test
  public void testDisallowValuesSetsDefaultValue() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("test_string")
        .getDisallowValuesBuilder()
        .addDisallowedValues(DISALLOWED_VALUE_1)
        .getUseDefaultBuilder();

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--test_string=" + DISALLOWED_VALUE_1);

    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(DISALLOWED_VALUE_1);

    enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder());

    testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(TestOptions.TEST_STRING_DEFAULT);
  }

  @Test
  public void testDisallowValuesSetsDefaultValueForRepeatableFlag() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("test_multiple_string")
        .getDisallowValuesBuilder()
        .addDisallowedValues(DISALLOWED_VALUE_1)
        .getUseDefaultBuilder();

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--test_multiple_string=" + DISALLOWED_VALUE_1);

    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testMultipleString).containsExactly(DISALLOWED_VALUE_1);

    enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder());

    testOptions = getTestOptions();
    // Default for repeatable flags is always empty.
    assertThat(testOptions.testMultipleString).isEmpty();
  }

  @Test
  public void testDisallowValuesRaisesErrorIfDefaultIsDisallowedAndSetsUseDefault()
      throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("test_string")
        .getDisallowValuesBuilder()
        .addDisallowedValues(TestOptions.TEST_STRING_DEFAULT)
        .getUseDefaultBuilder();

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);

    OptionsParsingException e =
        assertThrows(
            OptionsParsingException.class,
            () -> enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder()));
    assertThat(e).hasMessageThat().contains("but also specifies to use the default value");
  }
  
  @Test
  public void testDisallowValuesSetsNewValueWhenDefaultIsDisallowed() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("test_string")
        .getDisallowValuesBuilder()
        .addDisallowedValues(TestOptions.TEST_STRING_DEFAULT)
        .setNewValue(UNFILTERED_VALUE);

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);

    // Option should be the default since the use didn't specify a value.
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(TestOptions.TEST_STRING_DEFAULT);

    enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder());

    // Should now be "baz" because the policy set the new default to "baz"
    testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(UNFILTERED_VALUE);
  }

  @Test
  public void testDisallowValuesDisallowsFlagDefaultButNoPolicyDefault() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("test_string")
        .getDisallowValuesBuilder()
        // No new default is set
        .addDisallowedValues(TestOptions.TEST_STRING_DEFAULT);

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);

    // Option should be the default since the use didn't specify a value.
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(TestOptions.TEST_STRING_DEFAULT);

    assertThrows(
        OptionsParsingException.class,
        () -> enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder()));
  }

  @Test
  public void testDisallowValuesDisallowsListConverterFlag() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("test_list_converters")
        .getDisallowValuesBuilder()
        .addDisallowedValues("a");

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--test_list_converters=a,b,c");

    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testListConverters).isEqualTo(Arrays.asList("a", "b", "c"));

    OptionsParsingException e =
        assertThrows(
            OptionsParsingException.class,
            () -> enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder()));
    assertThat(e)
        .hasMessageThat()
        .contains(
            "Flag value 'a' for option '--test_list_converters' is not allowed by invocation "
                + "policy");
  }

  @Test
  public void testAllowValuesWithNullDefault_DoesNotConfuseNullForDefault() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("test_string_null_by_default")
        .setDisallowValues(
            DisallowValues.newBuilder()
                .addDisallowedValues("null")
                .setUseDefault(UseDefault.getDefaultInstance()));
    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    // Check the value before invocation policy enforcement.
    parser.parse("--test_string_null_by_default=null");
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testStringNullByDefault).isEqualTo("null");

    // Check the value afterwards.
    enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder());
    testOptions = getTestOptions();
    assertThat(testOptions.testStringNullByDefault).isNull();
  }
}
