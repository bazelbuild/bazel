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
    HtmlUtils.getUsageHtml(
        data.getOptionDefinitionFromName(fieldName), builder, HTML_ESCAPER, data, false, null);
    return builder.toString();
  }

  private String getHtmlUsageWithTags(String fieldName) {
    StringBuilder builder = new StringBuilder();
    HtmlUtils.getUsageHtml(
        data.getOptionDefinitionFromName(fieldName), builder, HTML_ESCAPER, data, true, null);
    return builder.toString();
  }

  private String getHtmlUsageWithCommandName(String fieldName, String commandName) {
    StringBuilder builder = new StringBuilder();
    HtmlUtils.getUsageHtml(
        data.getOptionDefinitionFromName(fieldName),
        builder,
        HTML_ESCAPER,
        data,
        false,
        commandName);
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
  public void commandNameAnchorId_htmlOutput() {
    assertThat(getHtmlUsageWithCommandName("test_string", "command_name"))
        .isEqualTo(
            """
            <dt id="command_name-flag--test_string"><code id="test_string"><a href="#command_name-flag--test_string">--test_string</a>=&lt;a string&gt;</code> default: "test string default"</dt>
            <dd>
            <p>a string-valued option to test simple option operations</p>
            </dd>
            """);
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
            """
            <dt id="flag--test_string"><code><a href="#flag--test_string">--test_string</a>=&lt;a string&gt;</code> default: "test string default"</dt>
            <dd>
            <p>a string-valued option to test simple option operations</p>
            </dd>
            """);
    assertThat(getHtmlUsageWithTags("test_string"))
        .isEqualTo(
            """
            <dt id="flag--test_string"><code><a href="#flag--test_string">--test_string</a>=&lt;a string&gt;</code> default: "test string default"</dt>
            <dd>
            <p>a string-valued option to test simple option operations</p>
            <p>Tags:
            <a href="#effect_tag_NO_OP"><code>no_op</code></a>
            </p></dd>
            """);
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
            """
            <dt id="flag--expanded_c"><code><a href="#flag--expanded_c">--expanded_c</a>=&lt;an integer&gt;</code> default: "12"</dt>
            <dd>
            <p>an int-value'd flag used to test expansion logic</p>
            </dd>
            """);
    assertThat(getHtmlUsageWithTags("expanded_c"))
        .isEqualTo(
            """
            <dt id="flag--expanded_c"><code><a href="#flag--expanded_c">--expanded_c</a>=&lt;an integer&gt;</code> default: "12"</dt>
            <dd>
            <p>an int-value'd flag used to test expansion logic</p>
            <p>Tags:
            <a href="#effect_tag_NO_OP"><code>no_op</code></a>
            </p></dd>
            """);
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
            """
            <dt id="flag--expanded_a"><code><a href="#flag--expanded_a">--[no]expanded_a</a></code> default: "true"</dt>
            <dd>
            <p>A boolean flag with unknown effect to test tagless usage text.</p>
            </dd>
            """);
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
            """
            <dt id="flag--test_multiple_string"><code><a href="#flag--test_multiple_string">--test_multiple_string</a>=&lt;a string&gt;</code> multiple uses are accumulated</dt>
            <dd>
            <p>a repeatable string-valued flag with its own unhelpful help text</p>
            </dd>
            """);
    assertThat(getHtmlUsageWithTags("test_multiple_string"))
        .isEqualTo(
            """
            <dt id="flag--test_multiple_string"><code><a href="#flag--test_multiple_string">--test_multiple_string</a>=&lt;a string&gt;</code> multiple uses are accumulated</dt>
            <dd>
            <p>a repeatable string-valued flag with its own unhelpful help text</p>
            <p>Tags:
            <a href="#effect_tag_NO_OP"><code>no_op</code></a>
            </p></dd>
            """);
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
            """
            <dt id="flag--test_list_converters"><code><a href="#flag--test_list_converters">--test_list_converters</a>=&lt;a list of strings&gt;</code> multiple uses are accumulated</dt>
            <dd>
            <p>a repeatable flag that accepts lists, but doesn't want to have lists of lists as a final type</p>
            </dd>
            """);
    assertThat(getHtmlUsageWithTags("test_list_converters"))
        .isEqualTo(
            """
            <dt id="flag--test_list_converters"><code><a href="#flag--test_list_converters">--test_list_converters</a>=&lt;a list of strings&gt;</code> multiple uses are accumulated</dt>
            <dd>
            <p>a repeatable flag that accepts lists, but doesn't want to have lists of lists as a final type</p>
            <p>Tags:
            <a href="#effect_tag_NO_OP"><code>no_op</code></a>
            </p></dd>
            """);
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
            """
            <dt id="flag--test_expansion"><code><a href="#flag--test_expansion">--test_expansion</a></code></dt>
            <dd>
            <p>this expands to an alphabet soup.</p>
            <p>Expands to:
            <br/>&nbsp;&nbsp;<code><a href="#flag--noexpanded_a">--noexpanded_a</a></code>
            <br/>&nbsp;&nbsp;<code><a href="#flag--expanded_b">--expanded_b=false</a></code>
            <br/>&nbsp;&nbsp;<code><a href="#flag--expanded_c">--expanded_c</a></code>
            <br/>&nbsp;&nbsp;<code><a href="#flag42">42</a></code>
            <br/>&nbsp;&nbsp;<code><a href="#flag--expanded_d">--expanded_d</a></code>
            <br/>&nbsp;&nbsp;<code><a href="#flagbar">bar</a></code>
            </p></dd>
            """);
    assertThat(getHtmlUsageWithTags("test_expansion"))
        .isEqualTo(
            """
            <dt id="flag--test_expansion"><code><a href="#flag--test_expansion">--test_expansion</a></code></dt>
            <dd>
            <p>this expands to an alphabet soup.</p>
            <p>Expands to:
            <br/>&nbsp;&nbsp;<code><a href="#flag--noexpanded_a">--noexpanded_a</a></code>
            <br/>&nbsp;&nbsp;<code><a href="#flag--expanded_b">--expanded_b=false</a></code>
            <br/>&nbsp;&nbsp;<code><a href="#flag--expanded_c">--expanded_c</a></code>
            <br/>&nbsp;&nbsp;<code><a href="#flag42">42</a></code>
            <br/>&nbsp;&nbsp;<code><a href="#flag--expanded_d">--expanded_d</a></code>
            <br/>&nbsp;&nbsp;<code><a href="#flagbar">bar</a></code>
            </p><p>Tags:
            <a href="#effect_tag_NO_OP"><code>no_op</code></a>
            </p></dd>
            """);
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
            """
            <dt id="flag--test_recursive_expansion_top_level"><code><a href="#flag--test_recursive_expansion_top_level">--test_recursive_expansion_top_level</a></code></dt>
            <dd>
            <p>Lets the children do all the work.</p>
            <p>Expands to:
            <br/>&nbsp;&nbsp;<code><a href="#flag--test_recursive_expansion_middle1">--test_recursive_expansion_middle1</a></code>
            <br/>&nbsp;&nbsp;<code><a href="#flag--test_recursive_expansion_middle2">--test_recursive_expansion_middle2</a></code>
            </p></dd>
            """);
    assertThat(getHtmlUsageWithTags("test_recursive_expansion_top_level"))
        .isEqualTo(
            """
            <dt id="flag--test_recursive_expansion_top_level"><code><a href="#flag--test_recursive_expansion_top_level">--test_recursive_expansion_top_level</a></code></dt>
            <dd>
            <p>Lets the children do all the work.</p>
            <p>Expands to:
            <br/>&nbsp;&nbsp;<code><a href="#flag--test_recursive_expansion_middle1">--test_recursive_expansion_middle1</a></code>
            <br/>&nbsp;&nbsp;<code><a href="#flag--test_recursive_expansion_middle2">--test_recursive_expansion_middle2</a></code>
            </p><p>Tags:
            <a href="#effect_tag_NO_OP"><code>no_op</code></a>
            </p></dd>
            """);
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
            """
            <dt id="flag--test_expansion_to_repeatable"><code><a href="#flag--test_expansion_to_repeatable">--test_expansion_to_repeatable</a></code></dt>
            <dd>
            <p>Go forth and multiply, they said.</p>
            <p>Expands to:
            <br/>&nbsp;&nbsp;<code><a href="#flag--test_multiple_string">--test_multiple_string=expandedFirstValue</a></code>
            <br/>&nbsp;&nbsp;<code><a href="#flag--test_multiple_string">--test_multiple_string=expandedSecondValue</a></code>
            </p></dd>
            """);
    assertThat(getHtmlUsageWithTags("test_expansion_to_repeatable"))
        .isEqualTo(
            """
            <dt id="flag--test_expansion_to_repeatable"><code><a href="#flag--test_expansion_to_repeatable">--test_expansion_to_repeatable</a></code></dt>
            <dd>
            <p>Go forth and multiply, they said.</p>
            <p>Expands to:
            <br/>&nbsp;&nbsp;<code><a href="#flag--test_multiple_string">--test_multiple_string=expandedFirstValue</a></code>
            <br/>&nbsp;&nbsp;<code><a href="#flag--test_multiple_string">--test_multiple_string=expandedSecondValue</a></code>
            </p><p>Tags:
            <a href="#effect_tag_NO_OP"><code>no_op</code></a>
            </p></dd>
            """);
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
            """
            <dt id="flag--test_implicit_requirement"><code><a href="#flag--test_implicit_requirement">--test_implicit_requirement</a>=&lt;a string&gt;</code> default: "direct implicit"</dt>
            <dd>
            <p>this option really needs that other one, isolation of purpose has failed.</p>
            </dd>
            """);
    assertThat(getHtmlUsageWithTags("test_implicit_requirement"))
        .isEqualTo(
            """
            <dt id="flag--test_implicit_requirement"><code><a href="#flag--test_implicit_requirement">--test_implicit_requirement</a>=&lt;a string&gt;</code> default: "direct implicit"</dt>
            <dd>
            <p>this option really needs that other one, isolation of purpose has failed.</p>
            <p>Tags:
            <a href="#effect_tag_NO_OP"><code>no_op</code></a>
            </p></dd>
            """);
  }

  @Test
  public void markdownInHelp_longTerminalOutput() {
    assertThat(getTerminalUsageWithoutTags("markdown_in_help", HelpVerbosity.LONG))
        .isEqualTo(
            """
              --markdown_in_help (a string; default: "default")
                normal, `code span`, *emphasis*, **strong emphasis**, [inline link](/url\s
                (title)), [reference link][ref]
                hard line\\
                break
                ```
                code block
                ```
                - unordered
                - list
                1. ordered
                2. list
               \s
                paragraph 1
               \s
                paragraph 2
               \s
                [ref]: /url (title)
               \s
            """);
  }

  @Test
  public void markdownInHelp_htmlOutput() {
    assertThat(getHtmlUsageWithoutTags("markdown_in_help"))
        .isEqualTo(
            """
            <dt id="flag--markdown_in_help"><code><a href="#flag--markdown_in_help">--markdown_in_help</a>=&lt;a string&gt;</code> default: "default"</dt>
            <dd>
            <p>normal, <code>code span</code>, <em>emphasis</em>, <strong>strong emphasis</strong>, <a href="/url" title="title">inline link</a>, <a href="/url" title="title">reference link</a>
            hard line<br />
            break</p>
            <pre><code>code block
            </code></pre>
            <ul>
            <li>unordered</li>
            <li>list</li>
            </ul>
            <ol>
            <li>ordered</li>
            <li>list</li>
            </ol>
            <p>paragraph 1</p>
            <p>paragraph 2</p>
            </dd>
            """);
  }
}
