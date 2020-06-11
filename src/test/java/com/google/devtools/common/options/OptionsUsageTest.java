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

import com.google.common.escape.Escaper;
import com.google.common.html.HtmlEscapers;
import com.google.devtools.common.options.OptionsParser.HelpVerbosity;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit test for {@link OptionsUsage}. */
@RunWith(JUnit4.class)
public final class OptionsUsageTest {

  private OptionsData data;
  private static final Escaper HTML_ESCAPER = HtmlEscapers.htmlEscaper();

  @Before
  public void setUp() {
    data = OptionsParser.getOptionsDataInternal(TestOptions.class);
  }

  private String getHtmlUsageWithoutTags(String fieldName) {
    StringBuilder builder = new StringBuilder();
    OptionsUsage.getUsageHtml(
        data.getOptionDefinitionFromName(fieldName), builder, HTML_ESCAPER, data, false);
    return builder.toString();
  }

  private String getHtmlUsageWithTags(String fieldName) {
    StringBuilder builder = new StringBuilder();
    OptionsUsage.getUsageHtml(
        data.getOptionDefinitionFromName(fieldName), builder, HTML_ESCAPER, data, true);
    return builder.toString();
  }

  private String getTerminalUsageWithoutTags(String fieldName, HelpVerbosity verbosity) {
    StringBuilder builder = new StringBuilder();
    OptionsUsage.getUsage(
        data.getOptionDefinitionFromName(fieldName), builder, verbosity, data, false);
    return builder.toString();
  }

  /**
   * Tests the future behavior of the options usage output. For short & medium verbosity, this
   * should be the same as the current default
   */
  private String getTerminalUsageWithTags(String fieldName, HelpVerbosity verbosity) {
    StringBuilder builder = new StringBuilder();
    OptionsUsage.getUsage(
        data.getOptionDefinitionFromName(fieldName), builder, verbosity, data, true);
    return builder.toString();
  }

  @Test
  public void stringValue_shortTerminalOutput() {
    assertThat(getTerminalUsageWithoutTags("test_string", HelpVerbosity.SHORT))
        .isEqualTo("  --test_string\n");
    assertThat(getTerminalUsageWithoutTags("test_string", HelpVerbosity.SHORT))
        .isEqualTo(getTerminalUsageWithTags("test_string", HelpVerbosity.SHORT));
  }

  @Test
  public void stringValue_mediumTerminalOutput() {
    assertThat(getTerminalUsageWithoutTags("test_string", HelpVerbosity.MEDIUM))
        .isEqualTo("  --test_string (a string; default: \"test string default\")\n");
    assertThat(getTerminalUsageWithoutTags("test_string", HelpVerbosity.MEDIUM))
        .isEqualTo(getTerminalUsageWithTags("test_string", HelpVerbosity.MEDIUM));
  }

  @Test
  public void stringValue_longTerminalOutput() {
    assertThat(getTerminalUsageWithoutTags("test_string", HelpVerbosity.LONG))
        .isEqualTo(
            "  --test_string (a string; default: \"test string default\")\n"
                + "    a string-valued option to test simple option operations\n");
    assertThat(getTerminalUsageWithTags("test_string", HelpVerbosity.LONG))
        .isEqualTo(
            "  --test_string (a string; default: \"test string default\")\n"
                + "    a string-valued option to test simple option operations\n"
                + "      Tags: no_op\n");
  }

  @Test
  public void stringValue_htmlOutput() {
    assertThat(getHtmlUsageWithoutTags("test_string"))
        .isEqualTo(
            "<dt id=\"flag--test_string\"><code><a href=\"#flag--test_string\">--test_string</a>"
                + "=&lt;a string&gt</code> default: \"test string default\"</dt>\n"
                + "<dd>\n"
                + "a string-valued option to test simple option operations\n"
                + "</dd>\n");
    assertThat(getHtmlUsageWithTags("test_string"))
        .isEqualTo(
            "<dt id=\"flag--test_string\"><code><a href=\"#flag--test_string\">--test_string</a>"
                + "=&lt;a string&gt</code> default: \"test string default\"</dt>\n"
                + "<dd>\n"
                + "a string-valued option to test simple option operations\n"
                + "<br>Tags: \n"
                + "<a href=\"#effect_tag_NO_OP\"><code>no_op</code></a>"
                + "</dd>\n");
  }

  @Test
  public void intValue_shortTerminalOutput() {
    assertThat(getTerminalUsageWithoutTags("expanded_c", HelpVerbosity.SHORT))
        .isEqualTo("  --expanded_c\n");
    assertThat(getTerminalUsageWithoutTags("expanded_c", HelpVerbosity.SHORT))
        .isEqualTo(getTerminalUsageWithTags("expanded_c", HelpVerbosity.SHORT));
  }

  @Test
  public void intValue_mediumTerminalOutput() {
    assertThat(getTerminalUsageWithoutTags("expanded_c", HelpVerbosity.MEDIUM))
        .isEqualTo("  --expanded_c (an integer; default: \"12\")\n");
    assertThat(getTerminalUsageWithoutTags("expanded_c", HelpVerbosity.MEDIUM))
        .isEqualTo(getTerminalUsageWithTags("expanded_c", HelpVerbosity.MEDIUM));
  }

  @Test
  public void intValue_longTerminalOutput() {
    assertThat(getTerminalUsageWithoutTags("expanded_c", HelpVerbosity.LONG))
        .isEqualTo(
            "  --expanded_c (an integer; default: \"12\")\n"
                + "    an int-value'd flag used to test expansion logic\n");
    assertThat(getTerminalUsageWithTags("expanded_c", HelpVerbosity.LONG))
        .isEqualTo(
            "  --expanded_c (an integer; default: \"12\")\n"
                + "    an int-value'd flag used to test expansion logic\n"
                + "      Tags: no_op\n");
  }

  @Test
  public void intValue_htmlOutput() {
    assertThat(getHtmlUsageWithoutTags("expanded_c"))
        .isEqualTo(
            "<dt id=\"flag--expanded_c\"><code>"
                + "<a href=\"#flag--expanded_c\">--expanded_c</a>"
                + "=&lt;an integer&gt</code> default: \"12\"</dt>\n"
                + "<dd>\n"
                + "an int-value&#39;d flag used to test expansion logic\n"
                + "</dd>\n");
    assertThat(getHtmlUsageWithTags("expanded_c"))
        .isEqualTo(
            "<dt id=\"flag--expanded_c\"><code>"
                + "<a href=\"#flag--expanded_c\">--expanded_c</a>"
                + "=&lt;an integer&gt</code> default: \"12\"</dt>\n"
                + "<dd>\n"
                + "an int-value&#39;d flag used to test expansion logic\n"
                + "<br>Tags: \n"
                + "<a href=\"#effect_tag_NO_OP\"><code>no_op</code></a>"
                + "</dd>\n");
  }

  @Test
  public void booleanValue_shortTerminalOutput() {
    assertThat(getTerminalUsageWithoutTags("expanded_a", HelpVerbosity.SHORT))
        .isEqualTo("  --[no]expanded_a\n");
    assertThat(getTerminalUsageWithoutTags("expanded_a", HelpVerbosity.SHORT))
        .isEqualTo(getTerminalUsageWithTags("expanded_a", HelpVerbosity.SHORT));
  }

  @Test
  public void booleanValue_mediumTerminalOutput() {
    assertThat(getTerminalUsageWithoutTags("expanded_a", HelpVerbosity.MEDIUM))
        .isEqualTo("  --[no]expanded_a (a boolean; default: \"true\")\n");
    assertThat(getTerminalUsageWithoutTags("expanded_a", HelpVerbosity.MEDIUM))
        .isEqualTo(getTerminalUsageWithTags("expanded_a", HelpVerbosity.MEDIUM));
  }

  @Test
  public void booleanValue_longTerminalOutput() {
    assertThat(getTerminalUsageWithoutTags("expanded_a", HelpVerbosity.LONG))
        .isEqualTo(
            "  --[no]expanded_a (a boolean; default: \"true\")\n"
                + "    A boolean flag with unknown effect to test tagless usage text.\n");
    // This flag has no useful tags, to verify that the tag line is omitted, so the usage line
    // should be the same in both tag and tag-free world.
    assertThat(getTerminalUsageWithoutTags("expanded_a", HelpVerbosity.LONG))
        .isEqualTo(getTerminalUsageWithTags("expanded_a", HelpVerbosity.LONG));
  }

  @Test
  public void booleanValue_htmlOutput() {
    assertThat(getHtmlUsageWithoutTags("expanded_a"))
        .isEqualTo(
            "<dt id=\"flag--expanded_a\"><code><a href=\"#flag--expanded_a\">"
                + "--[no]expanded_a</a></code> default: \"true\"</dt>\n"
                + "<dd>\n"
                + "A boolean flag with unknown effect to test tagless usage text.\n"
                + "</dd>\n");
    assertThat(getHtmlUsageWithoutTags("expanded_a")).isEqualTo(getHtmlUsageWithTags("expanded_a"));
  }

  @Test
  public void multipleValue_shortTerminalOutput() {
    assertThat(getTerminalUsageWithoutTags("test_multiple_string", HelpVerbosity.SHORT))
        .isEqualTo("  --test_multiple_string\n");
    assertThat(getTerminalUsageWithoutTags("test_multiple_string", HelpVerbosity.SHORT))
        .isEqualTo(getTerminalUsageWithTags("test_multiple_string", HelpVerbosity.SHORT));
  }

  @Test
  public void multipleValue_mediumTerminalOutput() {
    assertThat(getTerminalUsageWithoutTags("test_multiple_string", HelpVerbosity.MEDIUM))
        .isEqualTo("  --test_multiple_string (a string; may be used multiple times)\n");
    assertThat(getTerminalUsageWithoutTags("test_multiple_string", HelpVerbosity.MEDIUM))
        .isEqualTo(getTerminalUsageWithTags("test_multiple_string", HelpVerbosity.MEDIUM));
  }

  @Test
  public void multipleValue_longTerminalOutput() {
    assertThat(getTerminalUsageWithoutTags("test_multiple_string", HelpVerbosity.LONG))
        .isEqualTo(
            "  --test_multiple_string (a string; may be used multiple times)\n"
                + "    a repeatable string-valued flag with its own unhelpful help text\n");
    assertThat(getTerminalUsageWithTags("test_multiple_string", HelpVerbosity.LONG))
        .isEqualTo(
            "  --test_multiple_string (a string; may be used multiple times)\n"
                + "    a repeatable string-valued flag with its own unhelpful help text\n"
                + "      Tags: no_op\n");
  }

  @Test
  public void multipleValue_htmlOutput() {
    assertThat(getHtmlUsageWithoutTags("test_multiple_string"))
        .isEqualTo(
            "<dt id=\"flag--test_multiple_string\"><code>"
                + "<a href=\"#flag--test_multiple_string\">--test_multiple_string</a>"
                + "=&lt;a string&gt</code> "
                + "multiple uses are accumulated</dt>\n"
                + "<dd>\n"
                + "a repeatable string-valued flag with its own unhelpful help text\n"
                + "</dd>\n");
    assertThat(getHtmlUsageWithTags("test_multiple_string"))
        .isEqualTo(
            "<dt id=\"flag--test_multiple_string\"><code>"
                + "<a href=\"#flag--test_multiple_string\">--test_multiple_string</a>"
                + "=&lt;a string&gt</code> "
                + "multiple uses are accumulated</dt>\n"
                + "<dd>\n"
                + "a repeatable string-valued flag with its own unhelpful help text\n"
                + "<br>Tags: \n"
                + "<a href=\"#effect_tag_NO_OP\"><code>no_op</code></a>"
                + "</dd>\n");
  }

  @Test
  public void customConverterValue_shortTerminalOutput() {
    assertThat(getTerminalUsageWithoutTags("test_list_converters", HelpVerbosity.SHORT))
        .isEqualTo("  --test_list_converters\n");
    assertThat(getTerminalUsageWithoutTags("test_list_converters", HelpVerbosity.SHORT))
        .isEqualTo(getTerminalUsageWithTags("test_list_converters", HelpVerbosity.SHORT));
  }

  @Test
  public void customConverterValue_mediumTerminalOutput() {
    assertThat(getTerminalUsageWithoutTags("test_list_converters", HelpVerbosity.MEDIUM))
        .isEqualTo("  --test_list_converters (a list of strings; may be used multiple times)\n");
    assertThat(getTerminalUsageWithoutTags("test_list_converters", HelpVerbosity.MEDIUM))
        .isEqualTo(getTerminalUsageWithTags("test_list_converters", HelpVerbosity.MEDIUM));
  }

  @Test
  public void customConverterValue_longTerminalOutput() {
    assertThat(getTerminalUsageWithoutTags("test_list_converters", HelpVerbosity.LONG))
        .isEqualTo(
            "  --test_list_converters (a list of strings; may be used multiple times)\n"
                + "    a repeatable flag that accepts lists, but doesn't want to have lists of \n"
                + "    lists as a final type\n");
    assertThat(getTerminalUsageWithTags("test_list_converters", HelpVerbosity.LONG))
        .isEqualTo(
            "  --test_list_converters (a list of strings; may be used multiple times)\n"
                + "    a repeatable flag that accepts lists, but doesn't want to have lists of \n"
                + "    lists as a final type\n"
                + "      Tags: no_op\n");
  }

  @Test
  public void customConverterValue_htmlOutput() {
    assertThat(getHtmlUsageWithoutTags("test_list_converters"))
        .isEqualTo(
            "<dt id=\"flag--test_list_converters\"><code>"
                + "<a href=\"#flag--test_list_converters\">--test_list_converters</a>"
                + "=&lt;a list of strings&gt</code> "
                + "multiple uses are accumulated</dt>\n"
                + "<dd>\n"
                + "a repeatable flag that accepts lists, but doesn&#39;t want to have lists of "
                + "lists as a final type\n"
                + "</dd>\n");
    assertThat(getHtmlUsageWithTags("test_list_converters"))
        .isEqualTo(
            "<dt id=\"flag--test_list_converters\"><code>"
                + "<a href=\"#flag--test_list_converters\">--test_list_converters</a>"
                + "=&lt;a list of strings&gt</code> "
                + "multiple uses are accumulated</dt>\n"
                + "<dd>\n"
                + "a repeatable flag that accepts lists, but doesn&#39;t want to have lists of "
                + "lists as a final type\n"
                + "<br>Tags: \n"
                + "<a href=\"#effect_tag_NO_OP\"><code>no_op</code></a>"
                + "</dd>\n");
  }

  @Test
  public void staticExpansionOption_shortTerminalOutput() {
    assertThat(getTerminalUsageWithoutTags("test_expansion", HelpVerbosity.SHORT))
        .isEqualTo("  --test_expansion\n");
    assertThat(getTerminalUsageWithoutTags("test_expansion", HelpVerbosity.SHORT))
        .isEqualTo(getTerminalUsageWithTags("test_expansion", HelpVerbosity.SHORT));
  }

  @Test
  public void staticExpansionOption_mediumTerminalOutput() {
    assertThat(getTerminalUsageWithoutTags("test_expansion", HelpVerbosity.MEDIUM))
        .isEqualTo("  --test_expansion\n");
    assertThat(getTerminalUsageWithoutTags("test_expansion", HelpVerbosity.MEDIUM))
        .isEqualTo(getTerminalUsageWithTags("test_expansion", HelpVerbosity.MEDIUM));
  }

  @Test
  public void staticExpansionOption_longTerminalOutput() {
    assertThat(getTerminalUsageWithoutTags("test_expansion", HelpVerbosity.LONG))
        .isEqualTo(
            "  --test_expansion\n"
                + "    this expands to an alphabet soup.\n"
                + "      Expands to: --noexpanded_a --expanded_b=false --expanded_c 42 --\n"
                + "      expanded_d bar \n");
    assertThat(getTerminalUsageWithTags("test_expansion", HelpVerbosity.LONG))
        .isEqualTo(
            "  --test_expansion\n"
                + "    this expands to an alphabet soup.\n"
                + "      Expands to: --noexpanded_a --expanded_b=false --expanded_c 42 --\n"
                + "      expanded_d bar \n"
                + "      Tags: no_op\n");
  }

  @Test
  public void staticExpansionOption_htmlOutput() {
    assertThat(getHtmlUsageWithoutTags("test_expansion"))
        .isEqualTo(
            "<dt id=\"flag--test_expansion\"><code><a href=\"#flag--test_expansion\">"
                + "--test_expansion</a></code></dt>\n"
                + "<dd>\n"
                + "this expands to an alphabet soup.\n"
                + "<br/>\n"
                + "Expands to:<br/>\n"
                + "&nbsp;&nbsp;<code>"
                + "<a href=\"#flag--noexpanded_a\">--noexpanded_a</a></code><br/>\n"
                + "&nbsp;&nbsp;<code>"
                + "<a href=\"#flag--expanded_b\">--expanded_b=false</a></code><br/>\n"
                + "&nbsp;&nbsp;<code><a href=\"#flag--expanded_c\">--expanded_c</a></code><br/>\n"
                + "&nbsp;&nbsp;<code><a href=\"#flag42\">42</a></code><br/>\n"
                + "&nbsp;&nbsp;<code><a href=\"#flag--expanded_d\">--expanded_d</a></code><br/>\n"
                + "&nbsp;&nbsp;<code><a href=\"#flagbar\">bar</a></code><br/>\n"
                + "</dd>\n");
    assertThat(getHtmlUsageWithTags("test_expansion"))
        .isEqualTo(
            "<dt id=\"flag--test_expansion\"><code><a href=\"#flag--test_expansion\">"
                + "--test_expansion</a></code></dt>\n"
                + "<dd>\n"
                + "this expands to an alphabet soup.\n"
                + "<br/>\n"
                + "Expands to:<br/>\n"
                + "&nbsp;&nbsp;<code>"
                + "<a href=\"#flag--noexpanded_a\">--noexpanded_a</a></code><br/>\n"
                + "&nbsp;&nbsp;<code>"
                + "<a href=\"#flag--expanded_b\">--expanded_b=false</a></code><br/>\n"
                + "&nbsp;&nbsp;<code><a href=\"#flag--expanded_c\">--expanded_c</a></code><br/>\n"
                + "&nbsp;&nbsp;<code><a href=\"#flag42\">42</a></code><br/>\n"
                + "&nbsp;&nbsp;<code><a href=\"#flag--expanded_d\">--expanded_d</a></code><br/>\n"
                + "&nbsp;&nbsp;<code><a href=\"#flagbar\">bar</a></code><br/>\n"
                + "<br>Tags: \n"
                + "<a href=\"#effect_tag_NO_OP\"><code>no_op</code></a>"
                + "</dd>\n");
  }

  @Test
  public void recursiveExpansionOption_shortTerminalOutput() {
    assertThat(
            getTerminalUsageWithoutTags("test_recursive_expansion_top_level", HelpVerbosity.SHORT))
        .isEqualTo("  --test_recursive_expansion_top_level\n");
    assertThat(
            getTerminalUsageWithoutTags("test_recursive_expansion_top_level", HelpVerbosity.SHORT))
        .isEqualTo(
            getTerminalUsageWithTags("test_recursive_expansion_top_level", HelpVerbosity.SHORT));
  }

  @Test
  public void recursiveExpansionOption_mediumTerminalOutput() {
    assertThat(
            getTerminalUsageWithoutTags("test_recursive_expansion_top_level", HelpVerbosity.MEDIUM))
        .isEqualTo("  --test_recursive_expansion_top_level\n");
    assertThat(
            getTerminalUsageWithoutTags("test_recursive_expansion_top_level", HelpVerbosity.MEDIUM))
        .isEqualTo(
            getTerminalUsageWithTags("test_recursive_expansion_top_level", HelpVerbosity.MEDIUM));
  }

  @Test
  public void recursiveExpansionOption_longTerminalOutput() {
    assertThat(
            getTerminalUsageWithoutTags("test_recursive_expansion_top_level", HelpVerbosity.LONG))
        .isEqualTo(
            "  --test_recursive_expansion_top_level\n"
                + "    Lets the children do all the work.\n"
                + "      Expands to: --test_recursive_expansion_middle1 --\n"
                + "      test_recursive_expansion_middle2 \n");
    assertThat(getTerminalUsageWithTags("test_recursive_expansion_top_level", HelpVerbosity.LONG))
        .isEqualTo(
            "  --test_recursive_expansion_top_level\n"
                + "    Lets the children do all the work.\n"
                + "      Expands to: --test_recursive_expansion_middle1 --\n"
                + "      test_recursive_expansion_middle2 \n"
                + "      Tags: no_op\n");
  }

  @Test
  public void recursiveExpansionOption_htmlOutput() {
    assertThat(getHtmlUsageWithoutTags("test_recursive_expansion_top_level"))
        .isEqualTo(
            "<dt id=\"flag--test_recursive_expansion_top_level\"><code><a"
                + " href=\"#flag--test_recursive_expansion_top_level\">--test_recursive_expansion_top_level</a></code></dt>\n"
                + "<dd>\n"
                + "Lets the children do all the work.\n"
                + "<br/>\n"
                + "Expands to:<br/>\n"
                + "&nbsp;&nbsp;<code><a"
                + " href=\"#flag--test_recursive_expansion_middle1\">--test_recursive_expansion_middle1</a></code><br/>\n"
                + "&nbsp;&nbsp;<code><a"
                + " href=\"#flag--test_recursive_expansion_middle2\">--test_recursive_expansion_middle2</a></code><br/>\n"
                + "</dd>\n");
    assertThat(getHtmlUsageWithTags("test_recursive_expansion_top_level"))
        .isEqualTo(
            "<dt id=\"flag--test_recursive_expansion_top_level\"><code><a"
                + " href=\"#flag--test_recursive_expansion_top_level\">--test_recursive_expansion_top_level</a></code></dt>\n"
                + "<dd>\n"
                + "Lets the children do all the work.\n"
                + "<br/>\n"
                + "Expands to:<br/>\n"
                + "&nbsp;&nbsp;<code><a"
                + " href=\"#flag--test_recursive_expansion_middle1\">--test_recursive_expansion_middle1</a></code><br/>\n"
                + "&nbsp;&nbsp;<code><a"
                + " href=\"#flag--test_recursive_expansion_middle2\">--test_recursive_expansion_middle2</a></code><br/>\n"
                + "<br>Tags: \n"
                + "<a href=\"#effect_tag_NO_OP\"><code>no_op</code></a></dd>\n");
  }

  @Test
  public void expansionToMultipleValue_shortTerminalOutput() {
    assertThat(getTerminalUsageWithoutTags("test_expansion_to_repeatable", HelpVerbosity.SHORT))
        .isEqualTo("  --test_expansion_to_repeatable\n");
    assertThat(getTerminalUsageWithoutTags("test_expansion_to_repeatable", HelpVerbosity.SHORT))
        .isEqualTo(getTerminalUsageWithTags("test_expansion_to_repeatable", HelpVerbosity.SHORT));
  }

  @Test
  public void expansionToMultipleValue_mediumTerminalOutput() {
    assertThat(getTerminalUsageWithoutTags("test_expansion_to_repeatable", HelpVerbosity.MEDIUM))
        .isEqualTo("  --test_expansion_to_repeatable\n");
    assertThat(getTerminalUsageWithoutTags("test_expansion_to_repeatable", HelpVerbosity.MEDIUM))
        .isEqualTo(getTerminalUsageWithTags("test_expansion_to_repeatable", HelpVerbosity.MEDIUM));
  }

  @Test
  public void expansionToMultipleValue_longTerminalOutput() {
    assertThat(getTerminalUsageWithoutTags("test_expansion_to_repeatable", HelpVerbosity.LONG))
        .isEqualTo(
            "  --test_expansion_to_repeatable\n"
                + "    Go forth and multiply, they said.\n"
                + "      Expands to: --test_multiple_string=expandedFirstValue --\n"
                + "      test_multiple_string=expandedSecondValue \n");
    assertThat(getTerminalUsageWithTags("test_expansion_to_repeatable", HelpVerbosity.LONG))
        .isEqualTo(
            "  --test_expansion_to_repeatable\n"
                + "    Go forth and multiply, they said.\n"
                + "      Expands to: --test_multiple_string=expandedFirstValue --\n"
                + "      test_multiple_string=expandedSecondValue \n"
                + "      Tags: no_op\n");
  }

  @Test
  public void expansionToMultipleValue_htmlOutput() {
    assertThat(getHtmlUsageWithoutTags("test_expansion_to_repeatable"))
        .isEqualTo(
            "<dt id=\"flag--test_expansion_to_repeatable\"><code><a"
                + " href=\"#flag--test_expansion_to_repeatable\">--test_expansion_to_repeatable</a></code></dt>\n"
                + "<dd>\n"
                + "Go forth and multiply, they said.\n"
                + "<br/>\n"
                + "Expands to:<br/>\n"
                + "&nbsp;&nbsp;<code><a"
                + " href=\"#flag--test_multiple_string\">--test_multiple_string=expandedFirstValue</a></code><br/>\n"
                + "&nbsp;&nbsp;<code><a"
                + " href=\"#flag--test_multiple_string\">--test_multiple_string=expandedSecondValue</a></code><br/>\n"
                + "</dd>\n");
    assertThat(getHtmlUsageWithTags("test_expansion_to_repeatable"))
        .isEqualTo(
            "<dt id=\"flag--test_expansion_to_repeatable\"><code><a"
                + " href=\"#flag--test_expansion_to_repeatable\">--test_expansion_to_repeatable</a></code></dt>\n"
                + "<dd>\n"
                + "Go forth and multiply, they said.\n"
                + "<br/>\n"
                + "Expands to:<br/>\n"
                + "&nbsp;&nbsp;<code><a"
                + " href=\"#flag--test_multiple_string\">--test_multiple_string=expandedFirstValue</a></code><br/>\n"
                + "&nbsp;&nbsp;<code><a"
                + " href=\"#flag--test_multiple_string\">--test_multiple_string=expandedSecondValue</a></code><br/>\n"
                + "<br>Tags: \n"
                + "<a href=\"#effect_tag_NO_OP\"><code>no_op</code></a></dd>\n");
  }

  @Test
  public void implicitRequirementOption_shortTerminalOutput() {
    assertThat(getTerminalUsageWithoutTags("test_implicit_requirement", HelpVerbosity.SHORT))
        .isEqualTo("  --test_implicit_requirement\n");
    assertThat(getTerminalUsageWithoutTags("test_implicit_requirement", HelpVerbosity.SHORT))
        .isEqualTo(getTerminalUsageWithTags("test_implicit_requirement", HelpVerbosity.SHORT));
  }

  @Test
  public void implicitRequirementOption_mediumTerminalOutput() {
    assertThat(getTerminalUsageWithoutTags("test_implicit_requirement", HelpVerbosity.MEDIUM))
        .isEqualTo("  --test_implicit_requirement (a string; default: \"direct implicit\")\n");
    assertThat(getTerminalUsageWithoutTags("test_implicit_requirement", HelpVerbosity.MEDIUM))
        .isEqualTo(getTerminalUsageWithTags("test_implicit_requirement", HelpVerbosity.MEDIUM));
  }

  @Test
  public void implicitRequirementOption_longTerminalOutput() {
    assertThat(getTerminalUsageWithoutTags("test_implicit_requirement", HelpVerbosity.LONG))
        .isEqualTo(
            "  --test_implicit_requirement (a string; default: \"direct implicit\")\n"
                + "    this option really needs that other one, isolation of purpose has failed.\n"
                + "      Using this option will also add: --implicit_requirement_a=implicit \n"
                + "      requirement, required \n");
    assertThat(getTerminalUsageWithTags("test_implicit_requirement", HelpVerbosity.LONG))
        .isEqualTo(
            "  --test_implicit_requirement (a string; default: \"direct implicit\")\n"
                + "    this option really needs that other one, isolation of purpose has failed.\n"
                + "      Using this option will also add: --implicit_requirement_a=implicit \n"
                + "      requirement, required \n"
                + "      Tags: no_op\n");
  }

  @Test
  public void implicitRequirementOption_htmlOutput() {
    assertThat(getHtmlUsageWithoutTags("test_implicit_requirement"))
        .isEqualTo(
            "<dt id=\"flag--test_implicit_requirement\"><code>"
                + "<a href=\"#flag--test_implicit_requirement\">--test_implicit_requirement</a>"
                + "=&lt;a string&gt</code> "
                + "default: \"direct implicit\"</dt>\n"
                + "<dd>\n"
                + "this option really needs that other one, isolation of purpose has failed.\n"
                + "</dd>\n");
    assertThat(getHtmlUsageWithTags("test_implicit_requirement"))
        .isEqualTo(
            "<dt id=\"flag--test_implicit_requirement\"><code>"
                + "<a href=\"#flag--test_implicit_requirement\">--test_implicit_requirement</a>"
                + "=&lt;a string&gt</code> "
                + "default: \"direct implicit\"</dt>\n"
                + "<dd>\n"
                + "this option really needs that other one, isolation of purpose has failed.\n"
                + "<br>Tags: \n"
                + "<a href=\"#effect_tag_NO_OP\"><code>no_op</code></a>"
                + "</dd>\n");
  }

  @Test
  public void expansionFunctionOptionThatExpandsBasedOnOtherLoadedOptions_shortTerminalOutput() {
    assertThat(getTerminalUsageWithoutTags("prefix_expansion", HelpVerbosity.SHORT))
        .isEqualTo("  --prefix_expansion\n");
    assertThat(getTerminalUsageWithoutTags("prefix_expansion", HelpVerbosity.SHORT))
        .isEqualTo(getTerminalUsageWithTags("prefix_expansion", HelpVerbosity.SHORT));
  }

  @Test
  public void expansionFunctionOptionThatExpandsBasedOnOtherLoadedOptions_mediumTerminalOutput() {
    assertThat(getTerminalUsageWithoutTags("prefix_expansion", HelpVerbosity.MEDIUM))
        .isEqualTo("  --prefix_expansion\n");
    assertThat(getTerminalUsageWithoutTags("prefix_expansion", HelpVerbosity.MEDIUM))
        .isEqualTo(getTerminalUsageWithTags("prefix_expansion", HelpVerbosity.MEDIUM));
  }

  @Test
  public void expansionFunctionOptionThatExpandsBasedOnOtherLoadedOptions_longTerminalOutput() {
    assertThat(getTerminalUsageWithoutTags("prefix_expansion", HelpVerbosity.LONG))
        .isEqualTo(
            "  --prefix_expansion\n"
                + "    Expands to all options with a specific prefix.\n"
                + "      Expands to: --specialexp_bar --specialexp_foo \n");
    assertThat(getTerminalUsageWithTags("prefix_expansion", HelpVerbosity.LONG))
        .isEqualTo(
            "  --prefix_expansion\n"
                + "    Expands to all options with a specific prefix.\n"
                + "      Expands to: --specialexp_bar --specialexp_foo \n"
                + "      Tags: no_op\n");
  }

  @Test
  public void expansionFunctionOptionThatExpandsBasedOnOtherLoadedOptions_htmlOutput() {
    assertThat(getHtmlUsageWithoutTags("prefix_expansion"))
        .isEqualTo(
            "<dt id=\"flag--prefix_expansion\"><code>"
                + "<a href=\"#flag--prefix_expansion\">--prefix_expansion</a>"
                + "</code></dt>\n"
                + "<dd>\n"
                + "Expands to all options with a specific prefix.\n"
                + "<br/>\n"
                + "Expands to:<br/>\n"
                + "&nbsp;&nbsp;<code>"
                + "<a href=\"#flag--specialexp_bar\">--specialexp_bar</a></code><br/>\n"
                + "&nbsp;&nbsp;<code>"
                + "<a href=\"#flag--specialexp_foo\">--specialexp_foo</a></code><br/>\n"
                + "</dd>\n");
    assertThat(getHtmlUsageWithTags("prefix_expansion"))
        .isEqualTo(
            "<dt id=\"flag--prefix_expansion\"><code>"
                + "<a href=\"#flag--prefix_expansion\">--prefix_expansion</a>"
                + "</code></dt>\n"
                + "<dd>\n"
                + "Expands to all options with a specific prefix.\n"
                + "<br/>\n"
                + "Expands to:<br/>\n"
                + "&nbsp;&nbsp;<code>"
                + "<a href=\"#flag--specialexp_bar\">--specialexp_bar</a></code><br/>\n"
                + "&nbsp;&nbsp;<code>"
                + "<a href=\"#flag--specialexp_foo\">--specialexp_foo</a></code><br/>\n"
                + "<br>Tags: \n"
                + "<a href=\"#effect_tag_NO_OP\"><code>no_op</code></a>"
                + "</dd>\n");
  }

  @Test
  public void tagHeavyExpansionOption_shortTerminalOutput() {
    assertThat(getTerminalUsageWithoutTags("test_void_expansion_function", HelpVerbosity.SHORT))
        .isEqualTo("  --test_void_expansion_function\n");
    assertThat(getTerminalUsageWithoutTags("test_void_expansion_function", HelpVerbosity.SHORT))
        .isEqualTo(getTerminalUsageWithTags("test_void_expansion_function", HelpVerbosity.SHORT));
  }

  @Test
  public void tagHeavyExpansionOption_mediumTerminalOutput() {
    assertThat(getTerminalUsageWithoutTags("test_void_expansion_function", HelpVerbosity.MEDIUM))
        .isEqualTo("  --test_void_expansion_function\n");
    assertThat(getTerminalUsageWithoutTags("test_void_expansion_function", HelpVerbosity.MEDIUM))
        .isEqualTo(getTerminalUsageWithTags("test_void_expansion_function", HelpVerbosity.MEDIUM));
  }

  @Test
  public void tagHeavyExpansionOption_longTerminalOutput() {
    assertThat(getTerminalUsageWithoutTags("test_void_expansion_function", HelpVerbosity.LONG))
        .isEqualTo(
            "  --test_void_expansion_function\n"
                + "    Listing a ton of random tags to test the usage output.\n"
                + "      Expands to: --expanded_d void expanded \n");
    assertThat(getTerminalUsageWithTags("test_void_expansion_function", HelpVerbosity.LONG))
        .isEqualTo(
            "  --test_void_expansion_function\n"
                + "    Listing a ton of random tags to test the usage output.\n"
                + "      Expands to: --expanded_d void expanded \n"
                + "      Tags: action_command_lines, test_runner, terminal_output, experimental\n");
  }

  @Test
  public void tagHeavyExpansionOption_htmlOutput() {
    assertThat(getHtmlUsageWithoutTags("test_void_expansion_function"))
        .isEqualTo(
            "<dt id=\"flag--test_void_expansion_function\"><code><a"
                + " href=\"#flag--test_void_expansion_function\">--test_void_expansion_function</a></code></dt>\n"
                + "<dd>\n"
                + "Listing a ton of random tags to test the usage output.\n"
                + "<br/>\n"
                + "Expands to:<br/>\n"
                + "&nbsp;&nbsp;<code><a href=\"#flag--expanded_d\">--expanded_d</a></code><br/>\n"
                + "&nbsp;&nbsp;<code><a href=\"#flagvoid expanded\">void"
                + " expanded</a></code><br/>\n"
                + "</dd>\n");
    assertThat(getHtmlUsageWithTags("test_void_expansion_function"))
        .isEqualTo(
            "<dt id=\"flag--test_void_expansion_function\"><code><a"
                + " href=\"#flag--test_void_expansion_function\">--test_void_expansion_function</a></code></dt>\n"
                + "<dd>\n"
                + "Listing a ton of random tags to test the usage output.\n"
                + "<br/>\n"
                + "Expands to:<br/>\n"
                + "&nbsp;&nbsp;<code><a href=\"#flag--expanded_d\">--expanded_d</a></code><br/>\n"
                + "&nbsp;&nbsp;<code><a href=\"#flagvoid expanded\">void"
                + " expanded</a></code><br/>\n"
                + "<br>Tags: \n"
                + "<a href=\"#effect_tag_ACTION_COMMAND_LINES\"><code>action_command_lines</code></a>,"
                + " <a href=\"#effect_tag_TEST_RUNNER\"><code>test_runner</code></a>, <a"
                + " href=\"#effect_tag_TERMINAL_OUTPUT\"><code>terminal_output</code></a>, <a"
                + " href=\"#metadata_tag_EXPERIMENTAL\"><code>experimental</code></a></dd>\n");
  }
}
