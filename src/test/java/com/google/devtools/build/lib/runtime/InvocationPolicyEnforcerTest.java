// Copyright 2015 The Bazel Authors. All rights reserved.
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
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.InvocationPolicy;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.ByteArrayOutputStream;
import java.util.List;

@RunWith(JUnit4.class)
public class InvocationPolicyEnforcerTest {

  public static final String STRING_FLAG_DEFAULT = "test string default";
  
  public static class TestOptions extends OptionsBase {

    /*
     * Basic types
     */

    @Option(name = "test_string", defaultValue = STRING_FLAG_DEFAULT)
    public String testString;

    /*
     * Repeated flags
     */

    @Option(
        name = "test_multiple_string",
        defaultValue = "", // default value is ignored when allowMultiple = true.
        allowMultiple = true)
    public List<String> testMultipleString;

    /*
     * Expansion flags
     */

    @Option(
        name = "test_expansion",
        defaultValue = "null",
        expansion = {"--test_expansion_a", "--test_expansion_b", "--test_expansion_c"})
    public Void testExpansion;

    @Option(name = "test_expansion_a", defaultValue = "false")
    public boolean testExpansionA;

    @Option(name = "test_expansion_b", defaultValue = "false")
    public boolean testExpansionB;

    @Option(name = "test_expansion_c", defaultValue = "false")
    public boolean testExpansionC;

    /*
     * Implicit requirement flags
     */

    @Option(
        name = "test_implicit_requirement",
        defaultValue = "test implicit requirement default",
        implicitRequirements = {"--an_implicit_requirement=foo"})
    public String testImplicitRequirement;

    @Option(
        name = "an_implicit_requirement",
        defaultValue = "implicit default")
    public String anImplicitRequirement;

  }

  private static InvocationPolicyEnforcer createOptionsPolicyEnforcer(
      InvocationPolicy.Builder invocationPolicyBuilder) throws Exception {
    InvocationPolicy policyProto = invocationPolicyBuilder.build();

    // An OptionsPolicyEnforcer could be constructed in the test directly from the InvocationPolicy
    // proto, however Blaze will actually take the policy as another flag with a Base64 encoded
    // binary proto and parse that, so exercise that code path in the test.

    ByteArrayOutputStream out = new ByteArrayOutputStream();
    policyProto.writeTo(out);
    String policyBase64 = BaseEncoding.base64().encode(out.toByteArray());

    OptionsParser startupOptionsParser = OptionsParser.newOptionsParser(
        BlazeServerStartupOptions.class);
    String policyOption = "--invocation_policy=" + policyBase64;
    startupOptionsParser.parse(policyOption);

    return InvocationPolicyEnforcer.create(startupOptionsParser);
  }
  
  private OptionsParser parser;

  @Before
  public final void setParser() throws Exception  {
    parser = OptionsParser.newOptionsParser(TestOptions.class);
  }

  private TestOptions getTestOptions() {
    return parser.getOptions(TestOptions.class);
  }

  /*************************************************************************************************
   * Tests for SetValue
   ************************************************************************************************/

  /**
   * Tests that policy overrides a value when that value is from the user.
   */
  @Test
  public void testSetValueOverridesUser() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder.addFlagPoliciesBuilder()
        .setFlagName("test_string")
        .getSetValueBuilder()
            .addFlagValue("policy value");

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--test_string=user value");

    TestOptions testOptions = getTestOptions();
    assertEquals("user value", testOptions.testString);

    enforcer.enforce(parser, "build");

    // Get the options again after policy enforcement.
    testOptions = getTestOptions();
    assertEquals("policy value", testOptions.testString);
  }

  /**
   * Tests that policy overrides a value when the user doesn't specify the value (i.e., the value
   * is from the flag's default from its definition).
   */
  @Test
  public void testSetValueOverridesDefault() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder.addFlagPoliciesBuilder()
        .setFlagName("test_string")
        .getSetValueBuilder()
            .addFlagValue("policy value");

    // No user value.
    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);

    // All the flags should be their default value.
    TestOptions testOptions = getTestOptions();
    assertEquals(STRING_FLAG_DEFAULT, testOptions.testString);

    enforcer.enforce(parser, "build");

    // Get the options again after policy enforcement.
    testOptions = getTestOptions();
    assertEquals("policy value", testOptions.testString);
  }

  /**
   * Tests that SetValue overrides the user's value when the flag allows multiple values.
   */
  @Test
  public void testSetValueWithMultipleValuesOverridesUser() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder.addFlagPoliciesBuilder()
        .setFlagName("test_multiple_string")
        .getSetValueBuilder()
            .addFlagValue("policy value 1")
            .addFlagValue("policy value 2");

    InvocationPolicyEnforcer enforcer =  createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--test_multiple_string=user value 1", "--test_multiple_string=user value 2");

    // Options should not be modified by running the parser through OptionsPolicyEnforcer.create().
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testMultipleString)
        .containsExactly("user value 1", "user value 2").inOrder();
    //assertEquals(, testOptions.test_multiple_string);

    enforcer.enforce(parser, "build");

    // Get the options again after policy enforcement.
    testOptions = getTestOptions();
    assertThat(testOptions.testMultipleString)
        .containsExactly("policy value 1", "policy value 2").inOrder();
  }

  /**
   * Tests that policy overrides the default value when the flag allows multiple values and the user
   * doesn't provide a value.
   */
  @Test
  public void testSetValueWithMultipleValuesOverridesDefault() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder.addFlagPoliciesBuilder()
        .setFlagName("test_multiple_string")
        .getSetValueBuilder()
            .addFlagValue("policy value 1")
            .addFlagValue("policy value 2");

    // No user value.
    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);

    // Repeatable flags always default to the empty list.
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testMultipleString).isEmpty();

    enforcer.enforce(parser, "build");

    // Options should now be the values from the policy.
    testOptions = getTestOptions();
    assertThat(testOptions.testMultipleString)
        .containsExactly("policy value 1", "policy value 2").inOrder();
  }

  @Test
  public void testSetValueHasMultipleValuesButFlagIsNotMultiple() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder.addFlagPoliciesBuilder()
        .setFlagName("test_string") // Not repeatable flag.
        .getSetValueBuilder()
            .addFlagValue("policy value 1") // Has multiple values.
            .addFlagValue("policy value 2");

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);

    try {
      enforcer.enforce(parser, "build");
      fail();
    } catch (OptionsParsingException e) {
      // expected.
    }
  }

  @Test
  public void testSetValueWithExpansionFlags() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder.addFlagPoliciesBuilder()
        .setFlagName("test_expansion_b")
        .getSetValueBuilder()
            .addFlagValue("false");

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--test_expansion");

    // --test_expansion should turn on test_expansion a, b, and c
    TestOptions testOptions = getTestOptions();
    assertTrue(testOptions.testExpansionA);
    assertTrue(testOptions.testExpansionB);
    assertTrue(testOptions.testExpansionC);

    enforcer.enforce(parser, "build");

    // After policy enforcement, test_expansion_b should be set to false, but the
    // other two should remain the same.
    testOptions = getTestOptions();
    assertTrue(testOptions.testExpansionA);
    assertFalse(testOptions.testExpansionB);
    assertTrue(testOptions.testExpansionC);
  }

  @Test
  public void testSetValueWithImplicitlyRequiredFlags() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder.addFlagPoliciesBuilder()
        .setFlagName("an_implicit_requirement")
        .getSetValueBuilder()
            .addFlagValue("policy value");

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--test_implicit_requirement=user value");

    // test_implicit_requirement sets an_implicit_requirement to "foo"
    TestOptions testOptions = getTestOptions();
    assertEquals("user value", testOptions.testImplicitRequirement);
    assertEquals("foo", testOptions.anImplicitRequirement);

    enforcer.enforce(parser, "build");

    testOptions = getTestOptions();
    assertEquals("user value", testOptions.testImplicitRequirement);
    assertEquals("policy value", testOptions.anImplicitRequirement);
  }

  @Test
  public void testSetValueOverridable() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder.addFlagPoliciesBuilder()
        .setFlagName("test_string")
        .getSetValueBuilder()
            .addFlagValue("policy value")
            .setOverridable(true);

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--test_string=user value");

    // Repeatable flags always default to the empty list.
    TestOptions testOptions = getTestOptions();
    assertEquals("user value", testOptions.testString);

    enforcer.enforce(parser, "build");

    // Even though the policy sets the value for test_string, the policy is overridable and the
    // user set the value, so it should be the user's value.
    testOptions = getTestOptions();
    assertEquals("user value", testOptions.testString);
  }

  @Test
  public void testSetValueWithNoValueThrows() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder.addFlagPoliciesBuilder()
        .setFlagName("test_string")
        .getSetValueBuilder(); // No value.

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--test_string=user value");

    // Repeatable flags always default to the empty list.
    TestOptions testOptions = getTestOptions();
    assertEquals("user value", testOptions.testString);

    try {
      enforcer.enforce(parser, "build");
      fail();
    } catch (OptionsParsingException e) {
      // expected.
    }
  }

  /*************************************************************************************************
   * Tests for UseDefault
   ************************************************************************************************/

  @Test
  public void testUseDefault() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder.addFlagPoliciesBuilder()
        .setFlagName("test_string")
        .getUseDefaultBuilder();

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--test_string=user value");

    // Options should be the user specified value before enforcing policy.
    TestOptions testOptions = getTestOptions();
    assertEquals("user value", testOptions.testString);

    enforcer.enforce(parser, "build");

    // Get the options again after policy enforcement: The flag should now be back to its default
    // value
    testOptions = getTestOptions();
    assertEquals(STRING_FLAG_DEFAULT, testOptions.testString);
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
    assertEquals(STRING_FLAG_DEFAULT, testOptions.testString);

    enforcer.enforce(parser, "build");

    // Still the default.
    testOptions = getTestOptions();
    assertEquals(STRING_FLAG_DEFAULT, testOptions.testString);
  }

  @Test
  public void testUseDefaultWithExpansionFlags() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder.addFlagPoliciesBuilder()
        .setFlagName("test_expansion_b")
        .getUseDefaultBuilder();

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--test_expansion");

    // --test_expansion should turn on test_expansion a, b, and c
    TestOptions testOptions = getTestOptions();
    assertTrue(testOptions.testExpansionA);
    assertTrue(testOptions.testExpansionB);
    assertTrue(testOptions.testExpansionC);

    enforcer.enforce(parser, "build");

    // After policy enforcement, test_expansion_b should be back to its default (false), but the
    // other two should remain the same.
    testOptions = getTestOptions();
    assertTrue(testOptions.testExpansionA);
    assertFalse(testOptions.testExpansionB);
    assertTrue(testOptions.testExpansionC);
  }

  @Test
  public void testUseDefaultWithImplicitlyRequiredFlags() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder.addFlagPoliciesBuilder()
        .setFlagName("an_implicit_requirement")
        .getUseDefaultBuilder();

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--test_implicit_requirement=user value",
        "--an_implicit_requirement=implicit user value");

    // test_implicit_requirement sets an_implicit_requirement to "foo", which ignores the user's
    // value because the parser processes implicit values last.
    TestOptions testOptions = getTestOptions();
    assertEquals("user value", testOptions.testImplicitRequirement);
    assertEquals("foo", testOptions.anImplicitRequirement);

    // Then policy puts an_implicit_requirement back to its default.
    enforcer.enforce(parser, "build");

    testOptions = getTestOptions();
    assertEquals("user value", testOptions.testImplicitRequirement);
    assertEquals("implicit default", testOptions.anImplicitRequirement);
  }
  
  /*************************************************************************************************
   * Tests for AllowValues
   ************************************************************************************************/

  /**
   * Tests that AllowValues works in the normal case where the value the user specified is allowed
   * by the policy.
   */
  @Test
  public void testAllowValuesAllowsValue() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder.addFlagPoliciesBuilder()
        .setFlagName("test_string")
        .getAllowValuesBuilder()
            .addAllowedValues(STRING_FLAG_DEFAULT)
            .addAllowedValues("foo")
            .addAllowedValues("bar");

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--test_string=foo");

    // Option should be "foo" as specified by the user.
    TestOptions testOptions = getTestOptions();
    assertEquals("foo", testOptions.testString);

    enforcer.enforce(parser, "build");

    // Still "foo" since "foo" is allowed by the policy.
    testOptions = getTestOptions();
    assertEquals("foo", testOptions.testString);
  }

  @Test
  public void testAllowValuesDisallowsValue() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder.addFlagPoliciesBuilder()
        .setFlagName("test_string")
        .getAllowValuesBuilder()
            // no foo!
            .addAllowedValues(STRING_FLAG_DEFAULT)
            .addAllowedValues("bar");

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--test_string=foo");

    // Option should be "foo" as specified by the user.
    TestOptions testOptions = getTestOptions();
    assertEquals("foo", testOptions.testString);

    try {
      // Should throw because "foo" is not allowed.
      enforcer.enforce(parser, "build");
      fail();
    } catch (OptionsParsingException e) {
      // expected
    }
  }

  @Test
  public void testAllowValuesDisallowsMultipleValues() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder.addFlagPoliciesBuilder()
        .setFlagName("test_multiple_string")
        .getAllowValuesBuilder()
            .addAllowedValues("foo")
            .addAllowedValues("bar");

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--test_multiple_string=baz", "--test_multiple_string=bar");

    // Option should be "baz" and "bar" as specified by the user.
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testMultipleString).containsExactly("baz", "bar").inOrder();

    try {
      enforcer.enforce(parser, "build");
      fail();
    } catch (OptionsParsingException e) {
      // expected, since baz is not allowed.
    }
  }

  /**
   * Tests that AllowValues sets its default value when the user doesn't provide a value and the
   * flag's default value is disallowed.
   */
  @Test
  public void testAllowValuesSetsNewDefaultWhenFlagDefaultIsDisallowed() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder.addFlagPoliciesBuilder()
        .setFlagName("test_string")
        .getAllowValuesBuilder()
            // default value from flag's definition is not allowed
            .addAllowedValues("foo")
            .addAllowedValues("bar")
            .setNewDefaultValue("new default");

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);

    // Option should be its default
    TestOptions testOptions = getTestOptions();
    assertEquals(STRING_FLAG_DEFAULT, testOptions.testString);

    enforcer.enforce(parser, "build");

    // Flag's value should be the default value from the policy.
    testOptions = getTestOptions();
    assertEquals("new default", testOptions.testString);
  }

  @Test
  public void testAllowValuesDisallowsFlagDefaultButNoPolicyDefault() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder.addFlagPoliciesBuilder()
        .setFlagName("test_string")
        .getAllowValuesBuilder()
            // default value from flag's definition is not allowed, and no alternate default
            // is given.
            .addAllowedValues("foo")
            .addAllowedValues("bar");

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);

    // Option should be its default
    TestOptions testOptions = getTestOptions();
    assertEquals(STRING_FLAG_DEFAULT, testOptions.testString);

    try {
      enforcer.enforce(parser, "build");
      fail();
    } catch (OptionsParsingException e) {
      // expected.
    }
  }

  /*************************************************************************************************
   * Tests for DisallowValues
   ************************************************************************************************/

  @Test
  public void testDisallowValuesAllowsValue() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder.addFlagPoliciesBuilder()
        .setFlagName("test_string")
        .getDisallowValuesBuilder()
            .addDisallowedValues("foo")
            .addDisallowedValues("bar");

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--test_string=baz");

    // Option should be "baz" as specified by the user.
    TestOptions testOptions = getTestOptions();
    assertEquals("baz", testOptions.testString);

    enforcer.enforce(parser, "build");

    // Still "baz" since "baz" is allowed by the policy.
    testOptions = getTestOptions();
    assertEquals("baz", testOptions.testString);
  }

  @Test
  public void testDisallowValuesDisallowsValue() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder.addFlagPoliciesBuilder()
        .setFlagName("test_string")
        .getDisallowValuesBuilder()
            .addDisallowedValues("foo")
            .addDisallowedValues("bar");

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--test_string=foo");

    // Option should be "foo" as specified by the user.
    TestOptions testOptions = getTestOptions();
    assertEquals("foo", testOptions.testString);

    try {
      enforcer.enforce(parser, "build");
      fail();
    } catch (OptionsParsingException e) {
      // expected, since foo is disallowed.
    }
  }

  @Test
  public void testDisallowValuesDisallowsMultipleValues() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder.addFlagPoliciesBuilder()
        .setFlagName("test_multiple_string")
        .getDisallowValuesBuilder()
            .addDisallowedValues("foo")
            .addDisallowedValues("bar");

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--test_multiple_string=baz", "--test_multiple_string=bar");

    // Option should be "baz" and "bar" as specified by the user.
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testMultipleString).containsExactly("baz", "bar").inOrder();

    try {
      enforcer.enforce(parser, "build");
      fail();
    } catch (OptionsParsingException e) {
      // expected, since bar is disallowed.
    }
  }

  @Test
  public void testDisallowValuesSetsNewDefaultWhenFlagDefaultIsDisallowed() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder.addFlagPoliciesBuilder()
        .setFlagName("test_string")
        .getDisallowValuesBuilder()
            .addDisallowedValues(STRING_FLAG_DEFAULT)
            .setNewDefaultValue("baz");

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);

    // Option should be the default since the use didn't specify a value.
    TestOptions testOptions = getTestOptions();
    assertEquals(STRING_FLAG_DEFAULT, testOptions.testString);

    enforcer.enforce(parser, "build");

    // Should now be "baz" because the policy set the new default to "baz"
    testOptions = getTestOptions();
    assertEquals("baz", testOptions.testString);
  }

  @Test
  public void testDisallowValuesDisallowsFlagDefaultButNoPolicyDefault() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder.addFlagPoliciesBuilder()
        .setFlagName("test_string")
        .getDisallowValuesBuilder()
            // No new default is set
            .addDisallowedValues(STRING_FLAG_DEFAULT);

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);

    // Option should be the default since the use didn't specify a value.
    TestOptions testOptions = getTestOptions();
    assertEquals(STRING_FLAG_DEFAULT, testOptions.testString);

    try {
      enforcer.enforce(parser, "build");
      fail();
    } catch (OptionsParsingException e) {
      // expected.
    }
  }
  
  /*************************************************************************************************
   * Other tests
   ************************************************************************************************/

  @Test
  public void testFlagPolicyDoesNotApply() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder.addFlagPoliciesBuilder()
        .setFlagName("test_string")
        .addCommands("build")
        .getSetValueBuilder()
            .addFlagValue("policy value");

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--test_string=user value");

    TestOptions testOptions = getTestOptions();
    assertEquals("user value", testOptions.testString);

    enforcer.enforce(parser, "test");

    // Still user value.
    testOptions = getTestOptions();
    assertEquals("user value", testOptions.testString);
  }

  @Test
  public void testNonExistantFlagFromPolicy() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder.addFlagPoliciesBuilder()
        .setFlagName("i_do_not_exist")
        .getSetValueBuilder()
            .addFlagValue("policy value 1");
    invocationPolicyBuilder.addFlagPoliciesBuilder()
        .setFlagName("test_string")
        .getSetValueBuilder()
            .addFlagValue("policy value 2");
    
    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--test_string=user value");

    TestOptions testOptions = getTestOptions();
    assertEquals("user value", testOptions.testString);

    enforcer.enforce(parser, "test");

    // Still user value.
    testOptions = getTestOptions();
    assertEquals("policy value 2", testOptions.testString);
  }

  @Test
  public void testOperationNotSet() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder.addFlagPoliciesBuilder();
    // No operations added to the flag policy
    
    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--test_string=user value");
    
    TestOptions testOptions = getTestOptions();
    assertEquals("user value", testOptions.testString);

    // Shouldn't throw.
    enforcer.enforce(parser, "test");

    // Still user value.
    testOptions = getTestOptions();
    assertEquals("user value", testOptions.testString);
  }
}
