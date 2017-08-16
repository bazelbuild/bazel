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

import com.google.devtools.common.options.OptionsParser.HelpVerbosity;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit test for {@link OptionsUsage}. */
@RunWith(JUnit4.class)
public final class OptionsUsageTest {

  private OptionsData data;

  @Before
  public void setUp() {
    data = OptionsParser.getOptionsDataInternal(TestOptions.class);
  }

  private String getUsage(String fieldName, HelpVerbosity verbosity) {
    StringBuilder builder = new StringBuilder();
    OptionsUsage.getUsage(data.getFieldFromName(fieldName), builder, verbosity, data);
    return builder.toString();
  }

  @Test
  public void stringValue_short() {
    assertThat(getUsage("test_string", HelpVerbosity.SHORT)).isEqualTo("  --test_string\n");
  }

  @Test
  public void stringValue_medium() {
    assertThat(getUsage("test_string", HelpVerbosity.MEDIUM))
        .isEqualTo("  --test_string (a string; default: \"test string default\")\n");
  }

  @Test
  public void stringValue_long() {
    assertThat(getUsage("test_string", HelpVerbosity.LONG))
        .isEqualTo(
            "  --test_string (a string; default: \"test string default\")\n"
                + "    a string-valued option to test simple option operations\n");
  }

  @Test
  public void intValue_short() {
    assertThat(getUsage("expanded_c", HelpVerbosity.SHORT)).isEqualTo("  --expanded_c\n");
  }

  @Test
  public void intValue_medium() {
    assertThat(getUsage("expanded_c", HelpVerbosity.MEDIUM))
        .isEqualTo("  --expanded_c (an integer; default: \"12\")\n");
  }

  @Test
  public void intValue_long() {
    assertThat(getUsage("expanded_c", HelpVerbosity.LONG))
        .isEqualTo(
            "  --expanded_c (an integer; default: \"12\")\n"
                + "    an int-value'd flag used to test expansion logic\n");
  }

  @Test
  public void multipleValue_short() {
    assertThat(getUsage("test_multiple_string", HelpVerbosity.SHORT))
        .isEqualTo("  --test_multiple_string\n");
  }

  @Test
  public void multipleValue_medium() {
    assertThat(getUsage("test_multiple_string", HelpVerbosity.MEDIUM))
        .isEqualTo("  --test_multiple_string (a string; may be used multiple times)\n");
  }

  @Test
  public void multipleValue_long() {
    assertThat(getUsage("test_multiple_string", HelpVerbosity.LONG))
        .isEqualTo(
            "  --test_multiple_string (a string; may be used multiple times)\n"
                + "    a repeatable string-valued flag with its own unhelpful help text\n");
  }

  @Test
  public void customConverterValue_short() {
    assertThat(getUsage("test_list_converters", HelpVerbosity.SHORT))
        .isEqualTo("  --test_list_converters\n");
  }

  @Test
  public void customConverterValue_medium() {
    assertThat(getUsage("test_list_converters", HelpVerbosity.MEDIUM))
        .isEqualTo("  --test_list_converters (a list of strings; may be used multiple times)\n");
  }

  @Test
  public void customConverterValue_long() {
    assertThat(getUsage("test_list_converters", HelpVerbosity.LONG))
        .isEqualTo(
            "  --test_list_converters (a list of strings; may be used multiple times)\n"
                + "    a repeatable flag that accepts lists, but doesn't want to have lists of \n"
                + "    lists as a final type\n");
  }

  @Test
  public void staticExpansionOption_short() {
    assertThat(getUsage("test_expansion", HelpVerbosity.SHORT)).isEqualTo("  --test_expansion\n");
  }

  @Test
  public void staticExpansionOption_medium() {
    assertThat(getUsage("test_expansion", HelpVerbosity.MEDIUM)).isEqualTo("  --test_expansion\n");
  }

  @Test
  public void staticExpansionOption_long() {
    assertThat(getUsage("test_expansion", HelpVerbosity.LONG))
        .isEqualTo(
            "  --test_expansion\n"
                + "    this expands to an alphabet soup.\n"
                + "      Expands to: --noexpanded_a --expanded_b=false --expanded_c 42 --\n"
                + "      expanded_d bar \n");
  }

  @Test
  public void recursiveExpansionOption_short() {
    assertThat(getUsage("test_recursive_expansion_top_level", HelpVerbosity.SHORT))
        .isEqualTo("  --test_recursive_expansion_top_level\n");
  }

  @Test
  public void recursiveExpansionOption_medium() {
    assertThat(getUsage("test_recursive_expansion_top_level", HelpVerbosity.MEDIUM))
        .isEqualTo("  --test_recursive_expansion_top_level\n");
  }

  @Test
  public void recursiveExpansionOption_long() {
    assertThat(getUsage("test_recursive_expansion_top_level", HelpVerbosity.LONG))
        .isEqualTo(
            "  --test_recursive_expansion_top_level\n"
                + "    Lets the children do all the work.\n"
                + "      Expands to: --test_recursive_expansion_middle1 --\n"
                + "      test_recursive_expansion_middle2 \n");
  }

  @Test
  public void expansionToMultipleValue_short() {
    assertThat(getUsage("test_expansion_to_repeatable", HelpVerbosity.SHORT))
        .isEqualTo("  --test_expansion_to_repeatable\n");
  }

  @Test
  public void expansionToMultipleValue_medium() {
    assertThat(getUsage("test_expansion_to_repeatable", HelpVerbosity.MEDIUM))
        .isEqualTo("  --test_expansion_to_repeatable\n");
  }

  @Test
  public void expansionToMultipleValue_long() {
    assertThat(getUsage("test_expansion_to_repeatable", HelpVerbosity.LONG))
        .isEqualTo(
            "  --test_expansion_to_repeatable\n"
                + "    Go forth and multiply, they said.\n"
                + "      Expands to: --test_multiple_string=expandedFirstValue --\n"
                + "      test_multiple_string=expandedSecondValue \n");
  }

  @Test
  public void implicitRequirementOption_short() {
    assertThat(getUsage("test_implicit_requirement", HelpVerbosity.SHORT))
        .isEqualTo("  --test_implicit_requirement\n");
  }

  @Test
  public void implicitRequirementOption_medium() {
    assertThat(getUsage("test_implicit_requirement", HelpVerbosity.MEDIUM))
        .isEqualTo("  --test_implicit_requirement (a string; default: \"direct implicit\")\n");
  }

  @Test
  public void implicitRequirementOption_long() {
    assertThat(getUsage("test_implicit_requirement", HelpVerbosity.LONG))
        .isEqualTo(
            "  --test_implicit_requirement (a string; default: \"direct implicit\")\n"
                + "    this option really needs that other one, isolation of purpose has failed.\n"
                + "      Using this option will also add: --implicit_requirement_a=implicit \n"
                + "      requirement, required \n");
  }

  @Test
  public void expansionFunctionOptionThatReadsUserValue_short() {
    assertThat(getUsage("test_expansion_function", HelpVerbosity.SHORT))
        .isEqualTo("  --test_expansion_function\n");
  }

  @Test
  public void expansionFunctionOptionThatReadsUserValue_medium() {
    assertThat(getUsage("test_expansion_function", HelpVerbosity.MEDIUM))
        .isEqualTo("  --test_expansion_function\n");
  }

  @Test
  public void expansionFunctionOptionThatReadsUserValue_long() {
    assertThat(getUsage("test_expansion_function", HelpVerbosity.LONG))
        .isEqualTo(
            "  --test_expansion_function\n"
                + "    this is for testing expansion-by-function functionality.\n"
                + "      Expands to unknown options.\n");
  }

  @Test
  public void expansionFunctionOptionThatExpandsBasedOnOtherLoadedOptions_short() {
    assertThat(getUsage("prefix_expansion", HelpVerbosity.SHORT))
        .isEqualTo("  --prefix_expansion\n");
  }

  @Test
  public void expansionFunctionOptionThatExpandsBasedOnOtherLoadedOptions_medium() {
    assertThat(getUsage("prefix_expansion", HelpVerbosity.MEDIUM))
        .isEqualTo("  --prefix_expansion\n");
  }

  @Test
  public void expansionFunctionOptionThatExpandsBasedOnOtherLoadedOptions_long() {
    assertThat(getUsage("prefix_expansion", HelpVerbosity.LONG))
        .isEqualTo(
            "  --prefix_expansion\n"
                + "    Expands to all options with a specific prefix.\n"
                + "      Expands to: --specialexp_bar --specialexp_foo \n");
  }
}
