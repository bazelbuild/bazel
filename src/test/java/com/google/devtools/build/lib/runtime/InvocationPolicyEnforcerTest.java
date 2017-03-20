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
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableSet;
import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.flags.CommandNameCache;
import com.google.devtools.build.lib.flags.InvocationPolicyEnforcer;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.InvocationPolicy;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import java.io.ByteArrayOutputStream;
import java.util.Arrays;
import java.util.List;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class InvocationPolicyEnforcerTest {

  // Useful constants
  public static final String BUILD_COMMAND = "build";
  public static final String TEST_STRING_USER_VALUE = "user value";
  public static final String TEST_STRING_USER_VALUE_2 = "user value 2";
  public static final String TEST_STRING_POLICY_VALUE = "policy value";
  public static final String TEST_STRING_POLICY_VALUE_2 = "policy value 2";
  public static final String FILTERED_VALUE_1 = "foo";
  public static final String FILTERED_VALUE_2 = "bar";
  public static final String UNFILTERED_VALUE = "baz";

  /** Test converter that splits a string by commas to produce a list. */
  public static class ToListConverter implements Converter<List<String>> {

    public ToListConverter() {}

    @Override
    public List<String> convert(String input) throws OptionsParsingException {
      return Arrays.asList(input.split(","));
    }

    @Override
    public String getTypeDescription() {
      return "a list of strings";
    }
  }
  
  public static class TestOptions extends OptionsBase {

    /*
     * Basic types
     */

    public static final String TEST_STRING_DEFAULT = "test string default";
    @Option(name = "test_string", defaultValue = TEST_STRING_DEFAULT)
    public String testString;

    /*
     * Repeated flags
     */

    @Option(
      name = "test_multiple_string",
      defaultValue = "", // default value is ignored when allowMultiple=true.
      allowMultiple = true
    )
    public List<String> testMultipleString;

    /*
     * Flags with converters that return lists
     */

    @Option(
      name = "test_list_converters",
      defaultValue = "",
      allowMultiple = true,
      converter = ToListConverter.class
    )
    public List<String> testListConverters;

    /*
     * Expansion flags
     */

    public static final boolean EXPANDED_A_TEST_EXPANSION = false;
    public static final boolean EXPANDED_B_TEST_EXPANSION = false;
    public static final int EXPANDED_C_TEST_EXPANSION = 42;
    public static final String EXPANDED_D_TEST_EXPANSION = "bar";
    @Option(
      name = "test_expansion",
      defaultValue = "null",
      expansion = {
        "--noexpanded_a",
        "--expanded_b=false",
        "--expanded_c",
        "42",
        "--expanded_d",
        "bar"
      }
    )
    public Void testExpansion;

    public static final boolean EXPANDED_A_TEST_RECURSIVE_EXPANSION = false;
    public static final boolean EXPANDED_B_TEST_RECURSIVE_EXPANSION = false;
    public static final int EXPANDED_C_TEST_RECURSIVE_EXPANSION = 56;
    public static final String EXPANDED_D_TEST_RECURSIVE_EXPANSION = "baz";
    @Option(
        name = "test_recursive_expansion_top_level",
        defaultValue = "null",
        expansion = {
            "--test_recursive_expansion_middle1",
            "--test_recursive_expansion_middle2",
        })
    public Void testRecursiveExpansionTopLevel;

    @Option(
      name = "test_recursive_expansion_middle1",
      defaultValue = "null",
      expansion = {
        "--expanded_a=false",
        "--expanded_c=56",
      }
    )
    public Void testRecursiveExpansionMiddle1;

    @Option(
      name = "test_recursive_expansion_middle2",
      defaultValue = "null",
      expansion = {
        "--expanded_b=false",
        "--expanded_d=baz",
      }
    )
    public Void testRecursiveExpansionMiddle2;

    public static final boolean EXPANDED_A_DEFAULT = true;
    @Option(name = "expanded_a", defaultValue = "true")
    public boolean expandedA;

    public static final boolean EXPANDED_B_DEFAULT = true;
    @Option(name = "expanded_b", defaultValue = "true")
    public boolean expandedB;

    public static final int EXPANDED_C_DEFAULT = 12;
    @Option(name = "expanded_c", defaultValue = "12")
    public int expandedC;

    public static final String EXPANDED_D_DEFAULT = "foo";
    @Option(name = "expanded_d", defaultValue = "foo")
    public String expandedD;

    /*
     * Implicit requirement flags
     */

    public static final String TEST_IMPLICIT_REQUIREMENT_DEFAULT = "direct implicit";
    public static final String IMPLICIT_REQUIREMENT_A_REQUIRED = "implicit requirement, required";
    @Option(
      name = "test_implicit_requirement",
      defaultValue = TEST_IMPLICIT_REQUIREMENT_DEFAULT,
      implicitRequirements = {"--implicit_requirement_a=" + IMPLICIT_REQUIREMENT_A_REQUIRED}
    )
    public String testImplicitRequirement;

    public static final String IMPLICIT_REQUIREMENT_A_DEFAULT = "implicit requirement, unrequired";
    @Option(name = "implicit_requirement_a", defaultValue = IMPLICIT_REQUIREMENT_A_DEFAULT)
    public String implicitRequirementA;

    public static final String TEST_RECURSIVE_IMPLICIT_REQUIREMENT_DEFAULT = "recursive implicit";
    public static final String TEST_IMPLICIT_REQUIREMENT_REQUIRED = "intermediate, required";
    @Option(
      name = "test_recursive_implicit_requirement",
      defaultValue = TEST_RECURSIVE_IMPLICIT_REQUIREMENT_DEFAULT,
      implicitRequirements = {"--test_implicit_requirement=" + TEST_IMPLICIT_REQUIREMENT_REQUIRED}
    )
    public String testRecursiveImplicitRequirement;

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

    return InvocationPolicyEnforcer.create(
        startupOptionsParser.getOptions(BlazeServerStartupOptions.class).invocationPolicy);
  }
  
  private OptionsParser parser;

  @Before
  public final void setParser() throws Exception  {
    parser = OptionsParser.newOptionsParser(TestOptions.class);
  }

  @BeforeClass
  public static void setCommandNameCache() throws Exception {
    CommandNameCache.CommandNameCacheInstance.INSTANCE.setCommandNameCache(
        new CommandNameCache() {
          @Override
          public ImmutableSet<String> get(String commandName) {
            return ImmutableSet.of(commandName);
          }
        });
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
    parser.parse("--test_string=" + TEST_STRING_USER_VALUE);

    // Options should be the user specified value before enforcing policy.
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(TEST_STRING_USER_VALUE);

    enforcer.enforce(parser, BUILD_COMMAND);

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

    enforcer.enforce(parser, BUILD_COMMAND);

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

    enforcer.enforce(parser, BUILD_COMMAND);

    // After policy enforcement, all the flags that --test_expansion expanded into should be back
    // to their default values.
    testOptions = getTestOptions();
    assertThat(testOptions.expandedA).isEqualTo(TestOptions.EXPANDED_A_DEFAULT);
    assertThat(testOptions.expandedB).isEqualTo(TestOptions.EXPANDED_B_DEFAULT);
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

    enforcer.enforce(parser, BUILD_COMMAND);

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

    enforcer.enforce(parser, BUILD_COMMAND);

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
    enforcer.enforce(parser, BUILD_COMMAND);

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
        "--implicit_requirement_a=thrownaway value");

    // test_implicit_requirement sets implicit_requirement_a to "foo", which ignores the user's
    // value because the parser processes implicit values last.
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testImplicitRequirement).isEqualTo(TEST_STRING_USER_VALUE);
    assertThat(testOptions.implicitRequirementA)
        .isEqualTo(TestOptions.IMPLICIT_REQUIREMENT_A_REQUIRED);

    // Then policy puts implicit_requirement_a back to its default.
    enforcer.enforce(parser, BUILD_COMMAND);

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

    enforcer.enforce(parser, BUILD_COMMAND);

    // Policy enforcement should set everything back to its default value.
    testOptions = getTestOptions();
    assertThat(testOptions.testRecursiveImplicitRequirement)
        .isEqualTo(TestOptions.TEST_RECURSIVE_IMPLICIT_REQUIREMENT_DEFAULT);
    assertThat(testOptions.testImplicitRequirement)
        .isEqualTo(TestOptions.TEST_IMPLICIT_REQUIREMENT_DEFAULT);
    assertThat(testOptions.implicitRequirementA)
        .isEqualTo(TestOptions.IMPLICIT_REQUIREMENT_A_DEFAULT);
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
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("test_string")
        .getAllowValuesBuilder()
        .addAllowedValues(TestOptions.TEST_STRING_DEFAULT)
        .addAllowedValues(FILTERED_VALUE_1)
        .addAllowedValues(FILTERED_VALUE_2);

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--test_string=" + FILTERED_VALUE_1);

    // Option should be "foo" as specified by the user.
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(FILTERED_VALUE_1);

    enforcer.enforce(parser, BUILD_COMMAND);

    // Still "foo" since "foo" is allowed by the policy.
    testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(FILTERED_VALUE_1);
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
        .addAllowedValues(FILTERED_VALUE_2);

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--test_string=" + FILTERED_VALUE_1);

    // Option should be "foo" as specified by the user.
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(FILTERED_VALUE_1);

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
        .addAllowedValues(FILTERED_VALUE_1)
        .addAllowedValues(FILTERED_VALUE_2);

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse(
        "--test_multiple_string=" + UNFILTERED_VALUE, "--test_multiple_string=" + FILTERED_VALUE_2);

    // Option should be "baz" and "bar" as specified by the user.
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testMultipleString)
        .containsExactly(UNFILTERED_VALUE, FILTERED_VALUE_2)
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
        .addAllowedValues(FILTERED_VALUE_1)
        .addAllowedValues(FILTERED_VALUE_2)
        .setNewValue(FILTERED_VALUE_1);

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--test_string=" + UNFILTERED_VALUE);

    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(UNFILTERED_VALUE);

    enforcer.enforce(parser, BUILD_COMMAND);

    testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(FILTERED_VALUE_1);
  }

  @Test
  public void testAllowValuesSetsDefaultValue() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("test_string")
        .getAllowValuesBuilder()
        .addAllowedValues(FILTERED_VALUE_1)
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
        .addAllowedValues(FILTERED_VALUE_1)
        .addAllowedValues(FILTERED_VALUE_2)
        .getUseDefaultBuilder();

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse(
        "--test_multiple_string=" + FILTERED_VALUE_1, "--test_multiple_string=" + UNFILTERED_VALUE);

    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testMultipleString)
        .containsExactly(FILTERED_VALUE_1, UNFILTERED_VALUE)
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
        .addAllowedValues(FILTERED_VALUE_1)
        .addAllowedValues(FILTERED_VALUE_2)
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
        .addAllowedValues(FILTERED_VALUE_1)
        .addAllowedValues(FILTERED_VALUE_2);

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
  
  /*************************************************************************************************
   * Tests for DisallowValues
   ************************************************************************************************/

  @Test
  public void testDisallowValuesAllowsValue() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("test_string")
        .getDisallowValuesBuilder()
        .addDisallowedValues(FILTERED_VALUE_1)
        .addDisallowedValues(FILTERED_VALUE_2);

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--test_string=" + UNFILTERED_VALUE);

    // Option should be "baz" as specified by the user.
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(UNFILTERED_VALUE);

    enforcer.enforce(parser, BUILD_COMMAND);

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
        .addDisallowedValues(FILTERED_VALUE_1)
        .addDisallowedValues(FILTERED_VALUE_2);

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--test_string=" + FILTERED_VALUE_1);

    // Option should be "foo" as specified by the user.
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(FILTERED_VALUE_1);

    try {
      enforcer.enforce(parser, BUILD_COMMAND);
      fail();
    } catch (OptionsParsingException e) {
      // expected, since foo is disallowed.
    }
  }

  @Test
  public void testDisallowValuesDisallowsMultipleValues() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("test_multiple_string")
        .getDisallowValuesBuilder()
        .addDisallowedValues(FILTERED_VALUE_1)
        .addDisallowedValues(FILTERED_VALUE_2);

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse(
        "--test_multiple_string=" + UNFILTERED_VALUE, "--test_multiple_string=" + FILTERED_VALUE_2);

    // Option should be "baz" and "bar" as specified by the user.
    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testMultipleString)
        .containsExactly(UNFILTERED_VALUE, FILTERED_VALUE_2)
        .inOrder();

    try {
      enforcer.enforce(parser, BUILD_COMMAND);
      fail();
    } catch (OptionsParsingException e) {
      // expected, since bar is disallowed.
    }
  }

  @Test
  public void testDisallowValuesSetsNewValue() throws Exception {
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("test_string")
        .getDisallowValuesBuilder()
        .addDisallowedValues(FILTERED_VALUE_1)
        .setNewValue(UNFILTERED_VALUE);

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--test_string=" + FILTERED_VALUE_1);

    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(FILTERED_VALUE_1);

    enforcer.enforce(parser, BUILD_COMMAND);

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
        .addDisallowedValues(FILTERED_VALUE_1)
        .getUseDefaultBuilder();

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--test_string=" + FILTERED_VALUE_1);

    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(FILTERED_VALUE_1);

    enforcer.enforce(parser, BUILD_COMMAND);

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
        .addDisallowedValues(FILTERED_VALUE_1)
        .getUseDefaultBuilder();

    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--test_multiple_string=" + FILTERED_VALUE_1);

    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testMultipleString).containsExactly(FILTERED_VALUE_1);

    enforcer.enforce(parser, BUILD_COMMAND);

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

    try {
      enforcer.enforce(parser, BUILD_COMMAND);
      fail();
    } catch (OptionsParsingException e) {
      assertThat(e.getMessage()).contains("but also specifies to use the default value");
    }
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

    enforcer.enforce(parser, BUILD_COMMAND);

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

    try {
      enforcer.enforce(parser, BUILD_COMMAND);
      fail();
    } catch (OptionsParsingException e) {
      // expected.
    }
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

    try {
      enforcer.enforce(parser, BUILD_COMMAND);
      fail();
    } catch (OptionsParsingException e) {
      assertThat(e.getMessage())
          .contains(
              "Flag value 'a' for flag 'test_list_converters' is not allowed by invocation policy");
    }
  }
  
  /*************************************************************************************************
   * Other tests
   ************************************************************************************************/

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

    enforcer.enforce(parser, "test");

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
        .addFlagValue(TEST_STRING_POLICY_VALUE);
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("test_string")
        .getSetValueBuilder()
        .addFlagValue(TEST_STRING_POLICY_VALUE_2);
    
    InvocationPolicyEnforcer enforcer = createOptionsPolicyEnforcer(invocationPolicyBuilder);
    parser.parse("--test_string=" + TEST_STRING_USER_VALUE);

    TestOptions testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(TEST_STRING_USER_VALUE);

    enforcer.enforce(parser, "test");

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
    enforcer.enforce(parser, "test");

    // Still user value.
    testOptions = getTestOptions();
    assertThat(testOptions.testString).isEqualTo(TEST_STRING_USER_VALUE);
  }
}
