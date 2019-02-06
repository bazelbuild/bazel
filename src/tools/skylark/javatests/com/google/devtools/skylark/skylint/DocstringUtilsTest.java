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

package com.google.devtools.skylark.skylint;

import com.google.common.truth.Truth;
import com.google.devtools.skylark.common.DocstringUtils;
import com.google.devtools.skylark.common.DocstringUtils.DocstringInfo;
import com.google.devtools.skylark.common.DocstringUtils.DocstringParseError;
import com.google.devtools.skylark.common.DocstringUtils.ParameterDoc;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests the {@link DocstringUtils} class. */
@RunWith(JUnit4.class)
public class DocstringUtilsTest {
  @Test
  public void oneLineDocstring() throws Exception {
    List<DocstringParseError> errors = new ArrayList<>();
    DocstringInfo info = DocstringUtils.parseDocstring("summary", 0, errors);
    Truth.assertThat(info.getSummary()).isEqualTo("summary");
    Truth.assertThat(info.getParameters()).isEmpty();
    Truth.assertThat(info.getLongDescription()).isEmpty();
    Truth.assertThat(errors).isEmpty();
  }

  @Test
  public void missingBlankLineAfterSummary() throws Exception {
    List<DocstringParseError> errors = new ArrayList<>();
    DocstringInfo info = DocstringUtils.parseDocstring("summary\nfoo", 0, errors);
    Truth.assertThat(info.getSummary()).isEqualTo("summary");
    Truth.assertThat(info.getParameters()).isEmpty();
    Truth.assertThat(info.getLongDescription()).isEqualTo("foo");
    Truth.assertThat(errors.toString())
        .contains("2: the one-line summary should be followed by a blank line");
  }

  @Test
  public void multiParagraphDocstring() throws Exception {
    List<DocstringParseError> errors = new ArrayList<>();
    DocstringInfo info = DocstringUtils.parseDocstring("summary\n\nfoo\n\nbar\n", 0, errors);
    Truth.assertThat(info.getSummary()).isEqualTo("summary");
    Truth.assertThat(info.getParameters()).isEmpty();
    Truth.assertThat(info.getLongDescription()).isEqualTo("foo\n\nbar");
    Truth.assertThat(errors).isEmpty();
  }

  @Test
  public void inconsistentIndentation() throws Exception {
    List<DocstringParseError> errors = new ArrayList<>();
    DocstringInfo info =
        DocstringUtils.parseDocstring("summary\n" + "\n" + "  foo\n" + "bar\n  ", 2, errors);
    Truth.assertThat(info.getSummary()).isEqualTo("summary");
    Truth.assertThat(info.getParameters()).isEmpty();
    Truth.assertThat(info.getLongDescription()).isEqualTo("foo\nbar");
    Truth.assertThat(errors.toString())
        .contains("4: line indented too little (here: 0 spaces; expected: 2 spaces)");

    errors = new ArrayList<>();
    info =
        DocstringUtils.parseDocstring(
            "summary\n"
                + "\n"
                + "  baseline indentation\n"
                + "    more indentation can be useful (e.g. for example code)\n"
                + "  ",
            2,
            errors);
    Truth.assertThat(info.getSummary()).isEqualTo("summary");
    Truth.assertThat(info.getParameters()).isEmpty();
    Truth.assertThat(info.getLongDescription())
        .isEqualTo(
            "baseline indentation\n  more indentation can be useful (e.g. for example code)");
    Truth.assertThat(errors).isEmpty();
  }

  @Test
  public void inconsistentIndentationInSection() throws Exception {
    List<DocstringParseError> errors = new ArrayList<>();
    DocstringInfo info =
        DocstringUtils.parseDocstring(
            "summary\n"
                + "\n"
                + "Returns:\n"
                + " only one space indentation\n"
                + "unindented line after",
            0,
            errors);
    Truth.assertThat(info.getReturns()).isEqualTo("only one space indentation");
    Truth.assertThat(info.getLongDescription()).isEqualTo("unindented line after");
    Truth.assertThat(errors.toString())
        .contains(
            "4: text in a section has to be indented by two spaces"
                + " (relative to the left margin of the docstring)");
    Truth.assertThat(errors.toString()).contains("5: end of section without blank line");
  }

  @Test
  public void inconsistentIndentationInParameters() throws Exception {
    List<DocstringParseError> errors = new ArrayList<>();
    DocstringInfo info =
        DocstringUtils.parseDocstring(
            "summary\n"
                + "\n"
                + "Args:\n"
                + "  param0: two spaces indentation\n"
                + " param1: only one space indentation\n"
                + "  only two spaces indentation in continued line\n"
                + "unindented line after",
            0,
            errors);
    Truth.assertThat(info.getSummary()).isEqualTo("summary");
    Truth.assertThat(info.getParameters()).hasSize(2);
    ParameterDoc param0 = info.getParameters().get(0);
    Truth.assertThat(param0.getDescription()).isEqualTo("two spaces indentation");
    ParameterDoc param1 = info.getParameters().get(1);
    Truth.assertThat(param1.getDescription())
        .isEqualTo("only one space indentation\n only two spaces indentation in continued line");
    Truth.assertThat(errors.toString())
        .contains("5: inconsistent indentation of parameter lines (before: 2; here: 1 spaces)");
    Truth.assertThat(errors.toString()).contains("7: end of 'Args' section without blank line");
  }

  @Test
  public void whitespaceOnlyLinesCountAsBlank() throws Exception {
    List<DocstringParseError> errors = new ArrayList<>();
    DocstringInfo info =
        DocstringUtils.parseDocstring(
            "summary\n"
                + "        \n" // if not blank, would have too much indent
                + "    description\n"
                + "  \n"       // if not blank, would have too little indent
                + "    Args:\n"
                + "      foo: foo description\n"
                + "      \n"   // if not blank, would be indented just the right amount to end param
                               //   description but not Args section
                + "    Returns:\n"
                + "      returns description\n"
                + "     \n"    // if not blank, would be section content that's indented too little
                + "    ",
            4,
            errors);
    Truth.assertThat(info.getParameters()).hasSize(1);
    Truth.assertThat(info.getParameters().get(0).getDescription()).isEqualTo("foo description");
    Truth.assertThat(info.getReturns()).isEqualTo("returns description");
    Truth.assertThat(errors).isEmpty();
  }

  @Test
  public void closingQuoteMustBeProperlyIndented() throws Exception {
    List<DocstringParseError> errors = new ArrayList<>();
    DocstringUtils.parseDocstring("summary", 4, errors);
    Truth.assertThat(errors).isEmpty();

    errors = new ArrayList<>();
    DocstringUtils.parseDocstring(
        "summary\n"
            + "\n"
            + "    more description",
        4, errors);
    Truth.assertThat(errors.toString())
        .contains("3: closing docstring quote should be on its own line, indented the same as the "
            + "opening quote");

    errors = new ArrayList<>();
    DocstringUtils.parseDocstring(
        "summary\n"
            + "\n"
            + "    more description\n"
            + "  ",      // too little
        4, errors);
    Truth.assertThat(errors.toString())
        .contains("4: closing docstring quote should be indented the same as the opening quote");

    errors = new ArrayList<>();
    DocstringUtils.parseDocstring(
        "summary\n"
            + "\n"
            + "    more description\n"
            + "      ",  // too much
        4, errors);
    Truth.assertThat(errors.toString())
        .contains("4: closing docstring quote should be indented the same as the opening quote");

    errors = new ArrayList<>();
    DocstringUtils.parseDocstring(
        "summary\n"
            + "\n"
            + "    more description\n"
            + "",        // too little (empty line special case)
        4, errors);
    Truth.assertThat(errors.toString())
        .contains("4: closing docstring quote should be indented the same as the opening quote");
  }

  @Test
  public void emptySection() throws Exception {
    List<DocstringParseError> errors = new ArrayList<>();
    DocstringUtils.parseDocstring(
        "summary\n" + "\n" + "Args:\n" + "More description.\n", 0, errors);
    Truth.assertThat(errors.toString()).contains("3: section is empty or badly formatted");

    errors = new ArrayList<>();
    DocstringUtils.parseDocstring(
        "summary\n" + "\n" + "Returns:\n" + "More description\n", 0, errors);
    Truth.assertThat(errors.toString()).contains("3: section is empty");

    errors = new ArrayList<>();
    DocstringUtils.parseDocstring(
        "summary\n" + "\n" + "Deprecated:\n" + "More description\n", 0, errors);
    Truth.assertThat(errors.toString()).contains("3: section is empty");
  }

  @Test
  public void emptyParamDescription() throws Exception {
    List<DocstringParseError> errors = new ArrayList<>();
    DocstringUtils.parseDocstring("summary\n" + "\n" + "Args:\n" + "" + "  foo: \n\n", 0, errors);
    Truth.assertThat(errors.toString()).contains("4: empty parameter description for 'foo'");
  }

  @Test
  public void sectionOnOneLine() throws Exception {
    List<DocstringParseError> errors = new ArrayList<>();
    DocstringUtils.parseDocstring("summary\n" + "\n" + "Returns: foo\n", 0, errors);
    Truth.assertThat(errors).hasSize(1);
    Truth.assertThat(errors.get(0).toString())
        .startsWith("3: the return value should be documented in a section");
  }

  @Test
  public void docstringReturn() throws Exception {
    List<DocstringParseError> errors = new ArrayList<>();
    DocstringInfo info =
        DocstringUtils.parseDocstring(
            "summary\n"
                + "\n"
                + "  Returns:\n"
                + "    line 1\n"
                + "\n"
                + "    line 2\n"
                + "  ",
            2,
            errors);
    Truth.assertThat(info.getSummary()).isEqualTo("summary");
    Truth.assertThat(info.getReturns()).isEqualTo("line 1\n\nline 2");
    Truth.assertThat(info.getLongDescription()).isEmpty();
    Truth.assertThat(errors).isEmpty();
  }

  @Test
  public void docstringDeprecated() throws Exception {
    List<DocstringParseError> errors = new ArrayList<>();
    DocstringInfo info =
        DocstringUtils.parseDocstring(
            "summary\n"
                + "\n"
                + "  Deprecated:\n"
                + "    line 1\n"
                + "\n"
                + "    line 2\n"
                + "  ",
            2,
            errors);
    Truth.assertThat(info.getSummary()).isEqualTo("summary");
    Truth.assertThat(info.getDeprecated()).isEqualTo("line 1\n\nline 2");
    Truth.assertThat(info.getLongDescription()).isEmpty();
    Truth.assertThat(errors).isEmpty();
  }

  @Test
  public void docstringDeprecatedTheWrongWay() throws Exception {
    List<DocstringParseError> errors = new ArrayList<>();
    DocstringInfo info =
        DocstringUtils.parseDocstring("summary\n" + "\n" + "  Deprecated: foo\n  ", 2, errors);
    Truth.assertThat(info.getSummary()).isEqualTo("summary");
    Truth.assertThat(info.getDeprecated()).isEqualTo("Deprecated: foo");
    Truth.assertThat(info.getLongDescription()).isEqualTo("Deprecated: foo");
    Truth.assertThat(errors).hasSize(1);
    Truth.assertThat(errors.get(0).toString())
        .startsWith(
            "3: use a 'Deprecated:' section for deprecations, similar to a 'Returns:' section");

    errors = new ArrayList<>();
    info = DocstringUtils.parseDocstring(
        "summary\n" + "\n" + "  This is DEPRECATED.\n  ",
        2,
        errors);
    Truth.assertThat(info.getSummary()).isEqualTo("summary");
    Truth.assertThat(info.getDeprecated()).isEqualTo("This is DEPRECATED.");
    Truth.assertThat(info.getLongDescription()).isEqualTo("This is DEPRECATED.");
    Truth.assertThat(errors).hasSize(1);
    Truth.assertThat(errors.get(0).toString())
        .startsWith(
            "3: use a 'Deprecated:' section for deprecations, similar to a 'Returns:' section");
  }

  @Test
  public void docstringParameters() throws Exception {
    List<DocstringParseError> errors = new ArrayList<>();
    DocstringInfo info =
        DocstringUtils.parseDocstring(
            "summary\n"
                + "\n"
                + "  Args:\n"
                + "    param1: multi-\n"
                + "      line\n"
                + "    param2 (mutable, unused): bar\n"
                + "    *args: args\n"
                + "    **kwargs: kwargs\n"
                + "  ",
            2,
            errors);
    Truth.assertThat(info.getSummary()).isEqualTo("summary");
    Truth.assertThat(info.getParameters()).hasSize(4);
    Truth.assertThat(info.getLongDescription()).isEmpty();
    ParameterDoc firstParam = info.getParameters().get(0);
    ParameterDoc secondParam = info.getParameters().get(1);
    ParameterDoc thirdParam = info.getParameters().get(2);
    ParameterDoc fourthParam = info.getParameters().get(3);

    Truth.assertThat(firstParam.getParameterName()).isEqualTo("param1");
    Truth.assertThat(firstParam.getAttributes()).isEmpty();
    Truth.assertThat(firstParam.getDescription()).isEqualTo("multi-\n  line");

    Truth.assertThat(secondParam.getParameterName()).isEqualTo("param2");
    Truth.assertThat(secondParam.getAttributes()).isEqualTo(Arrays.asList("mutable", "unused"));
    Truth.assertThat(secondParam.getDescription()).isEqualTo("bar");

    Truth.assertThat(thirdParam.getParameterName()).isEqualTo("*args");
    Truth.assertThat(thirdParam.getAttributes()).isEmpty();
    Truth.assertThat(thirdParam.getDescription()).isEqualTo("args");

    Truth.assertThat(fourthParam.getParameterName()).isEqualTo("**kwargs");
    Truth.assertThat(fourthParam.getAttributes()).isEmpty();
    Truth.assertThat(fourthParam.getDescription()).isEqualTo("kwargs");

    Truth.assertThat(errors).isEmpty();
  }

  @Test
  public void docstringParametersWithBlankLines() throws Exception {
    List<DocstringParseError> errors = new ArrayList<>();
    DocstringInfo info =
        DocstringUtils.parseDocstring(
            "summary\n"
                + "\n"
                + "  Args:\n"
                + "    param1: multi-\n"
                + "\n"
                + "      line\n"
                + "\n"
                + "    param2: foo\n"
                + "  ",
            2,
            errors);
    Truth.assertThat(info.getSummary()).isEqualTo("summary");
    Truth.assertThat(info.getParameters()).hasSize(2);
    Truth.assertThat(info.getLongDescription()).isEmpty();
    ParameterDoc firstParam = info.getParameters().get(0);
    ParameterDoc secondParam = info.getParameters().get(1);

    Truth.assertThat(firstParam.getParameterName()).isEqualTo("param1");
    Truth.assertThat(firstParam.getAttributes()).isEmpty();
    Truth.assertThat(firstParam.getDescription()).isEqualTo("multi-\n\n  line");

    Truth.assertThat(secondParam.getParameterName()).isEqualTo("param2");
    Truth.assertThat(secondParam.getAttributes()).isEmpty();
    Truth.assertThat(secondParam.getDescription()).isEqualTo("foo");

    Truth.assertThat(errors).isEmpty();
  }

  @Test
  public void sectionNotPrecededByNewline() throws Exception {
    List<DocstringParseError> errors = new ArrayList<>();
    DocstringUtils.parseDocstring("summary\n" + "\n" + "  foo\n" + "  Args:", 2, errors);
    Truth.assertThat(errors.toString()).contains("4: section should be preceded by a blank line");
    errors = new ArrayList<>();
    DocstringUtils.parseDocstring("summary\n" + "\n" + "  foo\n" + "  Returns:", 2, errors);
    Truth.assertThat(errors.toString()).contains("4: section should be preceded by a blank line");
    errors = new ArrayList<>();
    DocstringUtils.parseDocstring("summary\n" + "\n" + "  foo\n" + "  Deprecated:", 2, errors);
    Truth.assertThat(errors.toString()).contains("4: section should be preceded by a blank line");
  }

  @Test
  public void duplicatedSectionsError() throws Exception {
    List<DocstringParseError> errors = new ArrayList<>();
    DocstringInfo info =
        DocstringUtils.parseDocstring(
            "summary\n"
                + "\n"
                + "  Args:\n"
                + "    param1: foo\n"
                + "\n"
                + "  Args:\n"
                + "    param2: bar\n"
                + "\n"
                + "  Returns:\n"
                + "    foo\n"
                + "\n"
                + "  Returns:\n"
                + "    bar\n"
                + "\n"
                + "  Deprecated:\n"
                + "    foo\n"
                + "\n"
                + "  Deprecated:\n"
                + "    bar\n"
                + "\n"
                + "  description",
            2,
            errors);
    Truth.assertThat(info.getParameters()).hasSize(2);
    Truth.assertThat(errors.toString()).contains("6: duplicate 'Args:' section");
    Truth.assertThat(errors.toString()).contains("12: duplicate 'Returns:' section");
    Truth.assertThat(errors.toString()).contains("18: duplicate 'Deprecated:' section");
  }

  @Test
  public void sectionsInWrongOrderError() throws Exception {
    List<DocstringParseError> errors = new ArrayList<>();
    DocstringInfo info =
        DocstringUtils.parseDocstring(
            "summary\n"
                + "\n"
                + "  Deprecated:\n"
                + "    bar\n"
                + "\n"
                + "  Returns:\n"
                + "    foo\n"
                + "\n"
                + "  Args:\n"
                + "    param1: foo\n"
                + "\n"
                + "  description\n"
                + "  ",
            2,
            errors);
    Truth.assertThat(info.getSummary()).isEqualTo("summary");
    Truth.assertThat(info.getParameters()).hasSize(1);
    Truth.assertThat(info.getReturns()).isEqualTo("foo");
    Truth.assertThat(info.getDeprecated()).isEqualTo("bar");
    Truth.assertThat(info.getLongDescription()).isEqualTo("description");
    Truth.assertThat(errors.toString())
        .contains("9: 'Args:' section should go before the 'Returns:' section");
    Truth.assertThat(errors.toString())
        .contains("9: 'Args:' section should go before the 'Deprecated:' section");
    Truth.assertThat(errors.toString())
        .contains("6: 'Returns:' section should go before the 'Deprecated:' section");
    Truth.assertThat(errors.toString())
        .contains(("12: description body should go before the special sections"));
  }

  @Test
  public void noRepeatedErrorAboutWrongOrder() throws Exception {
    List<DocstringParseError> errors = new ArrayList<>();
    DocstringInfo info =
        DocstringUtils.parseDocstring(
            "summary\n"
                + "\n"
                + "  Args:\n"
                + "    param1: foo\n"
                + "\n"
                + "  line 1\n"
                + "  line 2\n"
                + "  ",
            2,
            errors);
    Truth.assertThat(info.getSummary()).isEqualTo("summary");
    Truth.assertThat(info.getParameters()).hasSize(1);
    Truth.assertThat(info.getLongDescription()).isEqualTo("line 1\nline 2");
    Truth.assertThat(errors).hasSize(1);
    Truth.assertThat(errors.get(0).toString())
        .isEqualTo("6: description body should go before the special sections");
  }

  @Test
  public void invalidParameterDoc() throws Exception {
    List<DocstringParseError> errors = new ArrayList<>();
    DocstringInfo info =
        DocstringUtils.parseDocstring(
            "summary\n"
                + "\n"
                + "  Args:\n"
                + "    invalid parameter doc\n"
                + "\n"
                + "  description",
            2,
            errors);
    Truth.assertThat(info.getParameters()).isEmpty();
    Truth.assertThat(errors.toString())
        .contains(
            "4: invalid parameter documentation"
                + " (expected format: \"parameter_name: documentation\").");
  }

  @Test
  public void parseErrorContainsCorrectLine() throws Exception {
    List<DocstringParseError> errors = new ArrayList<>();
    DocstringUtils.parseDocstring(
        "summary\n" + "\n" + "  Args:\n" + "    invalid parameter doc\n", 2, errors);
    Truth.assertThat(errors.get(0).getLine()).isEqualTo("    invalid parameter doc");
  }
}
