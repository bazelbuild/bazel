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
import com.google.devtools.build.lib.syntax.FunctionDefStatement;
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
        findIssues("# no module docstring", "def function():", "  return # no function docstring")
            .toString();
    Truth.assertThat(errorMessage)
        .contains("1:1-2:1: file has no module docstring [missing-docstring]");
    Truth.assertThat(errorMessage)
        .contains("2:1-3:2: function 'function' has no docstring [missing-docstring]");
    // This function has zero statements. We want to make sure the end location is set correctly:
    BuildFileAST ast = BuildFileAST.parseString(event -> {}, "def foo():");
    FunctionDefStatement funDefWithEmptySuite = (FunctionDefStatement) ast.getStatements().get(0);
    Truth.assertThat(funDefWithEmptySuite.getStatements()).isEmpty();
    Truth.assertThat(DocstringChecker.check(ast).toString())
        .contains("1:1-1:10: function 'foo' has no docstring [missing-docstring]");
  }

  @Test
  public void reportMissingParameterDocumentation() throws Exception {
    List<Issue> errors =
        findIssues(
            "\"\"\" module docstring \"\"\"",
            "def f(param1, param2):",
            "  \"\"\"summary",
            "",
            "  more description",
            "\"\"\"",
            "  pass");
    Truth.assertThat(errors).hasSize(1);
    Truth.assertThat(errors.toString())
        .contains(
            "3:3-6:3: incomplete docstring: the function parameters are not documented"
                + " [inconsistent-docstring]");
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
        .contains(
            "2:3-6:5: incomplete docstring: parameter 'foo' not documented"
                + " [inconsistent-docstring]");
    Truth.assertThat(errorMessage)
        .contains(
            "2:3-6:5: incomplete docstring: parameter 'baz' not documented"
                + " [inconsistent-docstring]");
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
            "2:3-8:5: inconsistent docstring: parameter 'foo' appears in docstring"
                + " but not in function signature [inconsistent-docstring]");
    Truth.assertThat(errorMessage)
        .contains(
            "2:3-8:5: inconsistent docstring: parameter 'baz' appears in docstring"
                + " but not in function signature [inconsistent-docstring]");
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
            "2:3-7:5:"
                + " inconsistent docstring: order of parameters differs from function signature\n"
                + "Declaration order:   p1, p2\n"
                + "Documentation order: p2, p1\n"
                + " [inconsistent-docstring]");
  }

  @Test
  public void reportInvalidDocstringFormat() throws Exception {
    String errorMessage = findIssues("\"\"\"summary", "missing blank line\"\"\"").toString();
    Truth.assertThat(errorMessage)
        .contains(
            "2:1-2:18: bad docstring format: "
                + "the one-line summary should be followed by a blank line [bad-docstring-format]");

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
            "5:1-5:29: bad docstring format: "
                + "line indented too little (here: 1 spaces; expected: 2 spaces)"
                + " [bad-docstring-format]");
  }

  @Test
  public void reportMissingReturnDocumentation() throws Exception {
    List<Issue> errors =
        findIssues(
            "\"\"\" module docstring \"\"\"",
            "def f():",
            "  \"\"\"summary",
            "",
            "  more description",
            "  \"\"\"",
            "  return True");
    Truth.assertThat(errors).hasSize(1);
    Truth.assertThat(errors.toString())
        .contains(
            "3:3-6:5: incomplete docstring: the return value is not documented"
                + " [inconsistent-docstring]");
  }

  @Test
  public void dontReportReturnDocumentationIfNoReturnValue() throws Exception {
    Truth.assertThat(
            findIssues(
                "\"\"\" module docstring \"\"\"",
                "def f():",
                "  \"\"\"summary",
                "",
                "  more description",
                "  \"\"\"",
                "  return"))
        .isEmpty();
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
  public void dontReportSingleLineDocstring() throws Exception {
    Truth.assertThat(
            findIssues(
                "\"\"\"module docstring\"\"\"",
                "def function(param1, param2):",
                "  \"\"\"single line docstring is fine\"\"\"",
                "  return True"))
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
  public void dontReportCorrectFunctionDocstring() throws Exception {
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
                "",
                "  Returns:",
                "    True",
                "  \"\"\"",
                "  return True"))
        .isEmpty();
  }
}
