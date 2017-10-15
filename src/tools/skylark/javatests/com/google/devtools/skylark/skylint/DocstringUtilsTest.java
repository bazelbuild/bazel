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
    DocstringInfo info = DocstringUtils.parseDocString("summary", errors);
    Truth.assertThat(info.summary).isEqualTo("summary");
    Truth.assertThat(info.parameters).isEmpty();
    Truth.assertThat(info.longDescription).isEmpty();
    Truth.assertThat(errors).isEmpty();
  }

  @Test
  public void missingBlankLineAfterSummary() throws Exception {
    List<DocstringParseError> errors = new ArrayList<>();
    DocstringInfo info = DocstringUtils.parseDocString("summary\nfoo", errors);
    Truth.assertThat(info.summary).isEqualTo("summary");
    Truth.assertThat(info.parameters).isEmpty();
    Truth.assertThat(info.longDescription).isEqualTo("foo");
    Truth.assertThat(errors.toString())
        .contains(":2: the one-line summary should be followed by a blank line");
  }

  @Test
  public void multiParagraphDocstring() throws Exception {
    List<DocstringParseError> errors = new ArrayList<>();
    DocstringInfo info = DocstringUtils.parseDocString("summary\n\nfoo\n\nbar", errors);
    Truth.assertThat(info.summary).isEqualTo("summary");
    Truth.assertThat(info.parameters).isEmpty();
    Truth.assertThat(info.longDescription).isEqualTo("foo\n\nbar");
    Truth.assertThat(errors).isEmpty();
  }

  @Test
  public void inconsistentIndentation() throws Exception {
    List<DocstringParseError> errors = new ArrayList<>();
    DocstringInfo info =
        DocstringUtils.parseDocString("summary\n" + "\n" + "  foo\n" + "bar", errors);
    Truth.assertThat(info.summary).isEqualTo("summary");
    Truth.assertThat(info.parameters).isEmpty();
    Truth.assertThat(info.longDescription).isEqualTo("foo\nbar");
    Truth.assertThat(errors.toString())
        .contains(":4: line indented too little (here: 0 spaces; before: 2 spaces)");

    errors = new ArrayList<>();
    info = DocstringUtils.parseDocString("summary\n" + "\n" + "  foo\n" + "    bar\n", errors);
    Truth.assertThat(info.summary).isEqualTo("summary");
    Truth.assertThat(info.parameters).isEmpty();
    Truth.assertThat(info.longDescription).isEqualTo("foo\n  bar");
    Truth.assertThat(errors.toString())
        .contains(":4: line indented too much (here: 4 spaces; expected: 2 spaces)");
  }

  @Test
  public void inconsistentIndentationInParameters() throws Exception {
    List<DocstringParseError> errors = new ArrayList<>();
    DocstringInfo info =
        DocstringUtils.parseDocString("summary\n" + "\n" + "  Args:\n" + "  param1: foo\n", errors);
    Truth.assertThat(info.summary).isEqualTo("summary");
    Truth.assertThat(info.parameters).hasSize(1);
    Truth.assertThat(errors.toString())
        .contains(":4: parameter lines have to be indented by two spaces");
  }

  @Test
  public void docstringParameters() throws Exception {
    List<DocstringParseError> errors = new ArrayList<>();
    DocstringInfo info =
        DocstringUtils.parseDocString(
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
    Truth.assertThat(firstParam.description).isEqualTo("multi-\nline");

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
  public void argsSectionNotPrecededByNewline() throws Exception {
    List<DocstringParseError> errors = new ArrayList<>();
    DocstringUtils.parseDocString("summary\n" + "\n" + "  foo\n" + "  Args:", errors);
    Truth.assertThat(errors.toString()).contains(":4: section should be preceded by a blank line");
  }

  @Test
  public void twoArgsSectionsError() throws Exception {
    List<DocstringParseError> errors = new ArrayList<>();
    DocstringInfo info =
        DocstringUtils.parseDocString(
            "summary\n"
                + "\n"
                + "  Args:\n"
                + "    param1: foo\n"
                + "\n"
                + "  Args:\n"
                + "    param2: bar\n"
                + "\n"
                + "  description",
            errors);
    Truth.assertThat(info.parameters).hasSize(2);
    Truth.assertThat(errors.toString()).contains(":6: parameters were already documented before");
  }

  @Test
  public void invalidParameterDoc() throws Exception {
    List<DocstringParseError> errors = new ArrayList<>();
    DocstringInfo info =
        DocstringUtils.parseDocString(
            "summary\n"
                + "\n"
                + "  Args:\n"
                + "    invalid parameter doc\n"
                + "\n"
                + "  description",
            errors);
    Truth.assertThat(info.parameters).isEmpty();
    Truth.assertThat(errors.toString()).contains(":4: invalid parameter documentation");
  }
}
