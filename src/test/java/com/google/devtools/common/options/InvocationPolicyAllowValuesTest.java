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
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.AllowValues;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.InvocationPolicy;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.UseDefault;
import java.util.Arrays;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test InvocationPolicies with the AllowValues operation. */
@RunWith(JUnit4.class)
public class InvocationPolicyAllowValuesTest extends InvocationPolicyEnforcerTestBase {

  // Useful constants
  public static final String BUILD_COMMAND = "build";
  public static final String ALLOWED_VALUE_1 = "foo";
  public static final String ALLOWED_VALUE_2 = "bar";
  public static final String UNFILTERED_VALUE = "baz";

  /**
   * Tests that AllowValues works in the normal case where the value the user specified is allowed
   * by the policy.
   */
  @Test
  public void testAllowValuesAllowsValue() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("test_string")
        .getAllowValuesBuilder()
        .addAllowedValues(TestOptions.TEST_STRING_DEFAULT)
        .addAllowedValues(ALLOWED_VALUE_1)
        .addAllowedValues(ALLOWED_VALUE_2);

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--test_string=" + ALLOWED_VALUE_1);

    // Option should be "foo" as specified by the user.
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(ALLOWED_VALUE_1);

    enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder());

    // Still "foo" since "foo" is allowed by the policy.
    testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(ALLOWED_VALUE_1);
  }

  @Test
  public void testAllowValuesDisallowsValue() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("test_string")
        .getAllowValuesBuilder()
        // no foo!
        .addAllowedValues(TestOptions.TEST_STRING_DEFAULT)
        .addAllowedValues(ALLOWED_VALUE_2);

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--test_string=" + ALLOWED_VALUE_1);

    // Option should be "foo" as specified by the user.
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(ALLOWED_VALUE_1);

    // Should throw because "foo" is not allowed.
    assertThrows(
        OptionsParsingException.class,
        () -> enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder()));
  }

  @Test
  public void testAllowValuesDisallowsMultipleValues() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("test_multiple_string")
        .getAllowValuesBuilder()
        .addAllowedValues(ALLOWED_VALUE_1)
        .addAllowedValues(ALLOWED_VALUE_2);

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse(
        "--test_multiple_string=" + UNFILTERED_VALUE, "--test_multiple_string=" + ALLOWED_VALUE_2);

    // Option should be "baz" and "bar" as specified by the user.
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testMultipleString)
        .containsExactly(UNFILTERED_VALUE, ALLOWED_VALUE_2)
        .inOrder();

    // expected, since baz is not allowed.
    assertThrows(
        OptionsParsingException.class,
        () -> enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder()));
  }

  @Test
  public void testAllowValuesSetsNewValue() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("test_string")
        .getAllowValuesBuilder()
        .addAllowedValues(ALLOWED_VALUE_1)
        .addAllowedValues(ALLOWED_VALUE_2)
        .setNewValue(ALLOWED_VALUE_1);

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--test_string=" + UNFILTERED_VALUE);

    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(UNFILTERED_VALUE);

    enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder());

    testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(ALLOWED_VALUE_1);
  }

  @Test
  public void testAllowValuesSetsDefaultValue() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("test_string")
        .getAllowValuesBuilder()
        .addAllowedValues(ALLOWED_VALUE_1)
        .addAllowedValues(TestOptions.TEST_STRING_DEFAULT)
        .getUseDefaultBuilder();

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--test_string=" + UNFILTERED_VALUE);

    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(UNFILTERED_VALUE);

    enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder());

    testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(TestOptions.TEST_STRING_DEFAULT);
  }

  @Test
  public void testAllowValuesSetsDefaultValueForRepeatableFlag() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("test_multiple_string")
        .getAllowValuesBuilder()
        .addAllowedValues(ALLOWED_VALUE_1)
        .addAllowedValues(ALLOWED_VALUE_2)
        .getUseDefaultBuilder();

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse(
        "--test_multiple_string=" + ALLOWED_VALUE_1, "--test_multiple_string=" + UNFILTERED_VALUE);

    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testMultipleString)
        .containsExactly(ALLOWED_VALUE_1, UNFILTERED_VALUE)
        .inOrder();

    enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder());

    testOptions = getTestOptions();
    // Default value for repeatable flags is always empty.
    assertThat(testOptions.testMultipleString).isEmpty();
  }

  /**
   * Tests that AllowValues sets its default value when the user doesn't provide a value and the
   * flag's default value is disallowed.
   */
  @Test
  public void testAllowValuesSetsNewDefaultWhenFlagDefaultIsDisallowed() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("test_string")
        .getAllowValuesBuilder()
        // default value from flag's definition is not allowed
        .addAllowedValues(ALLOWED_VALUE_1)
        .addAllowedValues(ALLOWED_VALUE_2)
        .setNewValue("new default");

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);

    // Option should be its default
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(TestOptions.TEST_STRING_DEFAULT);

    enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder());

    // Flag's value should be the default value from the policy.
    testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo("new default");
  }

  @Test
  public void testAllowValuesDisallowsFlagDefaultButNoPolicyDefault() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("test_string")
        .getAllowValuesBuilder()
        // default value from flag's definition is not allowed, and no alternate default
        // is given.
        .addAllowedValues(ALLOWED_VALUE_1)
        .addAllowedValues(ALLOWED_VALUE_2);

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);

    // Option should be its default
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(TestOptions.TEST_STRING_DEFAULT);

    assertThrows(
        OptionsParsingException.class,
        () -> enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder()));
  }

  @Test
  public void testAllowValuesDisallowsListConverterFlagValues() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("test_list_converters")
        .getAllowValuesBuilder()
        .addAllowedValues("a");

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
            "Flag value 'b' for option '--test_list_converters' is not allowed by invocation "
                + "policy");
  }

  @Test
  public void testAllowValuesWithNullDefault_AcceptedValue() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("test_string_null_by_default")
        .setAllowValues(
            AllowValues.newBuilder()
                .addAllowedValues("a")
                .setUseDefault(UseDefault.getDefaultInstance()));
    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    // Check the value before invocation policy enforcement.
    parser.parse("--test_string_null_by_default=a");
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testStringNullByDefault).isEqualTo("a");

    // Check the value afterwards.
    enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder());
    testOptions = getTestOptions();
    assertThat(testOptions.testStringNullByDefault).isEqualTo("a");
  }

  @Test
  public void testAllowValuesWithNullDefault_UsesNullDefaultToOverrideUnacceptedValue()
      throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("test_string_null_by_default")
        .setAllowValues(
            AllowValues.newBuilder()
                .addAllowedValues("a")
                .setUseDefault(UseDefault.getDefaultInstance()));
    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    // Check the value before invocation policy enforcement.
    parser.parse("--test_string_null_by_default=b");
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testStringNullByDefault).isEqualTo("b");

    // Check the value afterwards.
    enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder());
    testOptions = getTestOptions();
    assertThat(testOptions.testStringNullByDefault).isNull();
  }

  @Test
  public void testAllowValuesWithNullDefault_AllowsUnsetValue() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("test_string_null_by_default")
        .setAllowValues(
            AllowValues.newBuilder()
                .addAllowedValues("a")
                .setUseDefault(UseDefault.getDefaultInstance()));
    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    // Check the value before invocation policy enforcement.
    parser.parse();
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testStringNullByDefault).isNull();

    // Check the value afterwards.
    enforcer.enforce(parser, BUILD_COMMAND, ImmutableList.builder());
    testOptions = getTestOptions();
    assertThat(testOptions.testStringNullByDefault).isNull();
  }
}
