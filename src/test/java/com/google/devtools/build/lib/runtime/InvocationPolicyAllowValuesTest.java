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

    enforcer.enforce(parser, BUILD_COMMAND);

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

    try {
      // Should throw because "foo" is not allowed.
      enforcer.enforce(parser, BUILD_COMMAND);
      fail();
    } catch (OptionsParsingException e) {
      // expected
    }
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

    try {
      enforcer.enforce(parser, BUILD_COMMAND);
      fail();
    } catch (OptionsParsingException e) {
      // expected, since baz is not allowed.
    }
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

    enforcer.enforce(parser, BUILD_COMMAND);

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

    enforcer.enforce(parser, BUILD_COMMAND);

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

    enforcer.enforce(parser, BUILD_COMMAND);

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

    enforcer.enforce(parser, BUILD_COMMAND);

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

    try {
      enforcer.enforce(parser, BUILD_COMMAND);
      fail();
    } catch (OptionsParsingException e) {
      // expected.
    }
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

    try {
      enforcer.enforce(parser, BUILD_COMMAND);
      fail();
    } catch (OptionsParsingException e) {
      assertThat(e.getMessage())
          .contains(
              "Flag value 'b' for flag 'test_list_converters' is not allowed by invocation policy");
    }
  }
}
