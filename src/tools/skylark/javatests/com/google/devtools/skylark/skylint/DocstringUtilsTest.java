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
import com.google.devtools.skylark.skylint.DocstringUtils.DocstringInfo;
import com.google.devtools.skylark.skylint.DocstringUtils.DocstringParseError;
import com.google.devtools.skylark.skylint.DocstringUtils.ParameterDoc;
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
    Truth.assertThat(info.summary).isEqualTo("summary");
    Truth.assertThat(info.parameters).isEmpty();
    Truth.assertThat(info.longDescription).isEmpty();
    Truth.assertThat(errors).isEmpty();
  }

  @Test
  public void missingBlankLineAfterSummary() throws Exception {
    List<DocstringParseError> errors = new ArrayList<>();
    DocstringInfo info = DocstringUtils.parseDocstring("summary\nfoo", 0, errors);
    Truth.assertThat(info.summary).isEqualTo("summary");
    Truth.assertThat(info.parameters).isEmpty();
    Truth.assertThat(info.longDescription).isEqualTo("foo");
    Truth.assertThat(errors.toString())
        .contains("2: the one-line summary should be followed by a blank line");
  }

  @Test
  public void multiParagraphDocstring() throws Exception {
    List<DocstringParseError> errors = new ArrayList<>();
    DocstringInfo info = DocstringUtils.parseDocstring("summary\n\nfoo\n\nbar", 0, errors);
    Truth.assertThat(info.summary).isEqualTo("summary");
    Truth.assertThat(info.parameters).isEmpty();
    Truth.assertThat(info.longDescription).isEqualTo("foo\n\nbar");
    Truth.assertThat(errors).isEmpty();
  }

  @Test
  public void inconsistentIndentation() throws Exception {
    List<DocstringParseError> errors = new ArrayList<>();
    DocstringInfo info =
        DocstringUtils.parseDocstring("summary\n" + "\n" + "  foo\n" + "bar", 2, errors);
    Truth.assertThat(info.summary).isEqualTo("summary");
    Truth.assertThat(info.parameters).isEmpty();
    Truth.assertThat(info.longDescription).isEqualTo("foo\nbar");
    Truth.assertThat(errors.toString())
        .contains("4: line indented too little (here: 0 spaces; expected: 2 spaces)");

    errors = new ArrayList<>();
    info =
        DocstringUtils.parseDocstring(
            "summary\n"
                + "\n"
                + "  baseline indentation\n"
                + "    more indentation can be useful (e.g. for example code)\n",
            2,
            errors);
    Truth.assertThat(info.summary).isEqualTo("summary");
    Truth.assertThat(info.parameters).isEmpty();
    Truth.assertThat(info.longDescription)
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
    Truth.assertThat(info.returns).isEqualTo("only one space indentation");
    Truth.assertThat(info.longDescription).isEqualTo("unindented line after");
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
    Truth.assertThat(info.summary).isEqualTo("summary");
    Truth.assertThat(info.parameters).hasSize(2);
    ParameterDoc param0 = info.parameters.get(0);
    Truth.assertThat(param0.description).isEqualTo("two spaces indentation");
    ParameterDoc param1 = info.parameters.get(1);
    Truth.assertThat(param1.description)
        .isEqualTo("only one space indentation\n only two spaces indentation in continued line");
    Truth.assertThat(errors.toString())
        .contains("5: inconsistent indentation of parameter lines (before: 2; here: 1 spaces)");
    Truth.assertThat(errors.toString()).contains("7: end of 'Args' section without blank line");
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
                + "\n"
                + "  remaining description",
            2,
            errors);
    Truth.assertThat(info.summary).isEqualTo("summary");
    Truth.assertThat(info.returns).isEqualTo("line 1\n\nline 2");
    Truth.assertThat(info.longDescription).isEqualTo("remaining description");
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
                + "\n"
                + "  remaining description",
            2,
            errors);
    Truth.assertThat(info.summary).isEqualTo("summary");
    Truth.assertThat(info.deprecated).isEqualTo("line 1\n\nline 2");
    Truth.assertThat(info.longDescription).isEqualTo("remaining description");
    Truth.assertThat(errors).isEmpty();
  }

  @Test
  public void docstringDeprecatedTheWrongWay() throws Exception {
    List<DocstringParseError> errors = new ArrayList<>();
    DocstringInfo info =
        DocstringUtils.parseDocstring("summary\n" + "\n" + "  Deprecated: foo\n", 2, errors);
    Truth.assertThat(info.summary).isEqualTo("summary");
    Truth.assertThat(info.deprecated).isEqualTo("Deprecated: foo");
    Truth.assertThat(info.longDescription).isEqualTo("Deprecated: foo");
    Truth.assertThat(errors).hasSize(1);
    Truth.assertThat(errors.get(0).toString())
        .isEqualTo(
            "3: use a 'Deprecated:' section for deprecations, similar to a 'Returns:' section");

    errors = new ArrayList<>();
    info = DocstringUtils.parseDocstring("summary\n" + "\n" + "  This is DEPRECATED.\n", 2, errors);
    Truth.assertThat(info.summary).isEqualTo("summary");
    Truth.assertThat(info.deprecated).isEqualTo("This is DEPRECATED.");
    Truth.assertThat(info.longDescription).isEqualTo("This is DEPRECATED.");
    Truth.assertThat(errors).hasSize(1);
    Truth.assertThat(errors.get(0).toString())
        .isEqualTo(
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
                + "\n"
                + "  description",
            2,
            errors);
    Truth.assertThat(info.summary).isEqualTo("summary");
    Truth.assertThat(info.parameters).hasSize(4);
    Truth.assertThat(info.longDescription).isEqualTo("description");
    ParameterDoc firstParam = info.parameters.get(0);
    ParameterDoc secondParam = info.parameters.get(1);
    ParameterDoc thirdParam = info.parameters.get(2);
    ParameterDoc fourthParam = info.parameters.get(3);

    Truth.assertThat(firstParam.parameterName).isEqualTo("param1");
    Truth.assertThat(firstParam.attributes).isEmpty();
    Truth.assertThat(firstParam.description).isEqualTo("multi-\n  line");

    Truth.assertThat(secondParam.parameterName).isEqualTo("param2");
    Truth.assertThat(secondParam.attributes).isEqualTo(Arrays.asList("mutable", "unused"));
    Truth.assertThat(secondParam.description).isEqualTo("bar");

    Truth.assertThat(thirdParam.parameterName).isEqualTo("*args");
    Truth.assertThat(thirdParam.attributes).isEmpty();
    Truth.assertThat(thirdParam.description).isEqualTo("args");

    Truth.assertThat(fourthParam.parameterName).isEqualTo("**kwargs");
    Truth.assertThat(fourthParam.attributes).isEmpty();
    Truth.assertThat(fourthParam.description).isEqualTo("kwargs");

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
                + "\n"
                + "  description",
            2,
            errors);
    Truth.assertThat(info.summary).isEqualTo("summary");
    Truth.assertThat(info.parameters).hasSize(2);
    Truth.assertThat(info.longDescription).isEqualTo("description");
    ParameterDoc firstParam = info.parameters.get(0);
    ParameterDoc secondParam = info.parameters.get(1);

    Truth.assertThat(firstParam.parameterName).isEqualTo("param1");
    Truth.assertThat(firstParam.attributes).isEmpty();
    Truth.assertThat(firstParam.description).isEqualTo("multi-\n\n  line");

    Truth.assertThat(secondParam.parameterName).isEqualTo("param2");
    Truth.assertThat(secondParam.attributes).isEmpty();
    Truth.assertThat(secondParam.description).isEqualTo("foo");

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
    Truth.assertThat(info.parameters).hasSize(2);
    Truth.assertThat(errors.toString()).contains("6: parameters were already documented above");
    Truth.assertThat(errors.toString()).contains("12: return value was already documented above");
    Truth.assertThat(errors.toString())
        .contains("18: deprecation message was already documented above");
  }

  @Test
  public void sectionsInWrongOrderError() throws Exception {
    List<DocstringParseError> errors = new ArrayList<>();
    DocstringInfo info =
        DocstringUtils.parseDocstring(
            "summary\n"
                + "\n"
                + "  Returns:\n"
                + "    foo\n"
                + "\n"
                + "  Args:\n"
                + "    param1: foo\n"
                + "\n"
                + "  description",
            2,
            errors);
    Truth.assertThat(info.parameters).hasSize(1);
    Truth.assertThat(errors.toString())
        .contains("6: parameters should be documented before the return value");
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
    Truth.assertThat(info.parameters).isEmpty();
    Truth.assertThat(errors.toString()).contains("4: invalid parameter documentation");
  }

  @Test
  public void parseErrorContainsCorrectLine() throws Exception {
    List<DocstringParseError> errors = new ArrayList<>();
    DocstringUtils.parseDocstring(
        "summary\n" + "\n" + "  Args:\n" + "    invalid parameter doc\n", 2, errors);
    Truth.assertThat(errors.get(0).line).isEqualTo("    invalid parameter doc");
  }
}
