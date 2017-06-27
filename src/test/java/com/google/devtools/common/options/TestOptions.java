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

import com.google.common.collect.ImmutableList;
import com.google.devtools.common.options.InvocationPolicyEnforcerTestBase.ToListConverter;
import java.util.List;

/** Options for testing. */
public class TestOptions extends OptionsBase {

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
      }
  )
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
   * Expansion into repeatable flags.
   */

  public static final String EXPANDED_MULTIPLE_1 = "expandedFirstValue";
  public static final String EXPANDED_MULTIPLE_2 = "expandedSecondValue";

  @Option(name = "test_expansion_to_repeatable",
      defaultValue = "null",
      expansion = {
          "--test_multiple_string=expandedFirstValue",
          "--test_multiple_string=expandedSecondValue"
      }
  )
  public Void testExpansionToRepeatable;


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

  public static final String TEST_EXPANSION_FUNCTION_ACCEPTED_VALUE = "valueA";
  public static final String EXPANDED_D_EXPANSION_FUNCTION_VALUE = "expanded valueA";

  /** Used for testing an expansion flag that requires a value. */
  public static class TestExpansionFunction implements ExpansionFunction {
    @Override
    public ImmutableList<String> getExpansion(ExpansionContext expansionContext)
        throws OptionsParsingException {
      String value = expansionContext.getUnparsedValue();
      if (value == null) {
        throw new ExpansionNeedsValueException("Expansion value not set.");
      } else if (value.equals(TEST_EXPANSION_FUNCTION_ACCEPTED_VALUE)) {
        return ImmutableList.of("--expanded_d", EXPANDED_D_EXPANSION_FUNCTION_VALUE);
      } else {
        throw new OptionsParsingException("Unrecognized expansion value: " + value);
      }
    }
  }

  @Option(
    name = "test_expansion_function",
    defaultValue = "null",
    expansionFunction = TestExpansionFunction.class
  )
  public Void testExpansionFunction;

  public static final String EXPANDED_D_VOID_EXPANSION_FUNCTION_VALUE = "void expanded";

  /** Used for testing an expansion flag that doesn't requires a value. */
  public static class TestVoidExpansionFunction implements ExpansionFunction {
    @Override
    public ImmutableList<String> getExpansion(ExpansionContext expansionContext) {
      return ImmutableList.of("--expanded_d", EXPANDED_D_VOID_EXPANSION_FUNCTION_VALUE);
    }
  }

  @Option(
    name = "test_void_expansion_function",
    defaultValue = "null",
    expansionFunction = TestVoidExpansionFunction.class
  )
  public Void testVoidExpansionFunction;
}
