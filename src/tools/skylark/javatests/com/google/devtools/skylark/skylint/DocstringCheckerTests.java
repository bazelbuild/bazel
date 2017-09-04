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
import com.google.devtools.build.lib.syntax.BuildFileAST;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests the lint done by {@link DocstringChecker}. */
@RunWith(JUnit4.class)
public class DocstringCheckerTests {
  private static List<Issue> findIssues(String... lines) {
    String content = String.join("\n", lines);
    BuildFileAST ast =
        BuildFileAST.parseString(
            event -> {
              throw new IllegalArgumentException(event.getMessage());
            },
            content);
    return DocstringChecker.check(ast);
  }

  @Test
  public void reportMissingDocString() throws Exception {
    String errorMessage =
        findIssues("# no module docstring", "def function():", "  pass # no function docstring")
            .toString();
    Truth.assertThat(errorMessage).contains(":2:1: file has no module docstring");
    Truth.assertThat(errorMessage).contains(":2:1: function 'function' has no docstring");
  }

  @Test
  public void reportMissingParameterDocumentation() throws Exception {
    List<Issue> errors =
        findIssues(
            "\"\"\" module docstring \"\"\"",
            "def f(param1, param2):",
            "  \"\"\"summary",
            "",
            "more description",
            "\"\"\"",
            "  pass");
    Truth.assertThat(errors).hasSize(1);
    Truth.assertThat(errors.toString())
        .contains(":3:3: incomplete docstring: the function parameters are not documented");
  }

  @Test
  public void reportUndocumentedParameters() throws Exception {
    String errorMessage =
        findIssues(
                "def function(foo, bar, baz):",
                "  \"\"\"summary",
                "",
                "  Args:",
                "    bar: blabla",
                "  \"\"\"",
                "  pass")
            .toString();
    Truth.assertThat(errorMessage)
        .contains(":2:3: incomplete docstring: parameter 'foo' not documented");
    Truth.assertThat(errorMessage)
        .contains(":2:3: incomplete docstring: parameter 'baz' not documented");
  }

  @Test
  public void reportObsoleteParameterDocumentation() throws Exception {
    String errorMessage =
        findIssues(
                "def function(bar):",
                "  \"\"\"summary",
                "",
                "  Args:",
                "    foo: blabla",
                "    bar: blabla",
                "    baz: blabla",
                "  \"\"\"",
                "  pass")
            .toString();
    Truth.assertThat(errorMessage)
        .contains(
            ":2:3: inconsistent docstring: parameter 'foo' appears in docstring"
                + " but not in function signature");
    Truth.assertThat(errorMessage)
        .contains(
            ":2:3: inconsistent docstring: parameter 'baz' appears in docstring"
                + " but not in function signature");
  }

  @Test
  public void reportParametersDocumentedInDifferentOrder() throws Exception {
    String errorMessage =
        findIssues(
                "def function(p1, p2):",
                "  \"\"\"summary",
                "",
                "  Args:",
                "    p2: blabla",
                "    p1: blabla",
                "  \"\"\"",
                "  pass")
            .toString();
    Truth.assertThat(errorMessage)
        .contains(
            ":2:3: inconsistent docstring: order of parameters differs from function signature\n"
                + "Declaration order:   p1, p2\n"
                + "Documentation order: p2, p1");
  }

  @Test
  public void reportInvalidDocstringFormat() throws Exception {
    String errorMessage = findIssues("\"\"\"summary", "missing blank line\"\"\"").toString();
    Truth.assertThat(errorMessage)
        .contains(
            ":2:1: invalid docstring format: "
                + "the one-line summary should be followed by a blank line");

    errorMessage =
        findIssues(
                "def f():",
                "  \"\"\"summary",
                "",
                "  foo",
                " bad indentation in this line",
                "\"\"\"")
            .toString();
    Truth.assertThat(errorMessage)
        .contains(
            ":5:1: invalid docstring format: "
                + "line indented too little (here: 1 spaces; before: 2 spaces)");
  }

  @Test
  public void dontReportExistingDocstrings() throws Exception {
    Truth.assertThat(
            findIssues(
                "\"\"\"This is a module docstring",
                "\n\"\"\"",
                "def function():",
                "  \"\"\" This is a function docstring\n\"\"\""))
        .isEmpty();
  }

  @Test
  public void dontReportSummaryDocstringWithoutParameters() throws Exception {
    Truth.assertThat(
            findIssues(
                "\"\"\"module docstring\"\"\"",
                "def function(param1, param2):",
                "  \"\"\"summary without parameter docs is fine\"\"\""))
        .isEmpty();
  }

  @Test
  public void dontReportPrivateFunctionWithoutDocstring() throws Exception {
    Truth.assertThat(
            findIssues(
                "\"\"\" Module docstring\n\"\"\"",
                "def _private_function():",
                "  pass # no docstring necessary for private functions"))
        .isEmpty();
  }

  @Test
  public void dontReportFunctionDocstringWithCorrectParameters() throws Exception {
    Truth.assertThat(
            findIssues(
                "\"\"\" module docstring \"\"\"",
                "def function(param1, param2, *args, **kwargs):",
                "  \"\"\"summary",
                "",
                "  Args:",
                "    param1: foo",
                "    param2 (foo, bar): baz",
                "    *args: foo",
                "    **kwargs: bar",
                "  \"\"\"",
                "  pass"))
        .isEmpty();
  }
}
