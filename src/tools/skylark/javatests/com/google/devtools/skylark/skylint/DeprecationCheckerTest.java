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

/** Tests the lint done by {@link DeprecationChecker}. */
@RunWith(JUnit4.class)
public class DeprecationCheckerTest {
  private static List<Issue> findIssues(String... lines) {
    String content = String.join("\n", lines);
    BuildFileAST ast =
        BuildFileAST.parseString(
            event -> {
              throw new IllegalArgumentException(event.getMessage());
            },
            content);
    return DeprecationChecker.check(ast);
  }

  @Test
  public void symbolDeprecatedInSameFile() {
    String errorMessages =
        findIssues(
                "def f():",
                "  g()",
                "  h()",
                "  print(x)",
                "def g():",
                "  '''Foo.",
                "  ",
                "  Deprecated:",
                "    Reason.'''",
                "def h():",
                "  '''Bar.",
                "  ",
                "  This function is DEPRECATED for some reason.",
                "  The deprecation should really be documented in a 'Deprecated:' section",
                "  but the linter should recognize this kind of deprecation as well'''",
                "x = 0",
                "'''A deprecated variable.",
                "",
                "Deprecated:",
                "  Reason.",
                "'''")
            .toString();
    Truth.assertThat(errorMessages)
        .contains("2:3: usage of 'g' is deprecated: Reason. [deprecated-symbol]");
    Truth.assertThat(errorMessages)
        .contains(
            "3:3: usage of 'h' is deprecated: This function is DEPRECATED for some reason."
                + " [deprecated-symbol]");
    Truth.assertThat(errorMessages)
        .contains("4:9: usage of 'x' is deprecated: Reason. [deprecated-symbol]");
  }

  @Test
  public void deprecatedFunctionAliasing() {
    String errorMessages =
        findIssues(
                "def f():",
                "  h = g",
                "  h()",
                "def g():",
                "  '''Foo.",
                "  ",
                "  Deprecated:",
                "    reason'''")
            .toString();
    Truth.assertThat(errorMessages)
        .contains("2:7: usage of 'g' is deprecated: reason [deprecated-symbol]");
  }

  @Test
  public void functionNotDeprecated() {
    Truth.assertThat(
            findIssues(
                "def f():",
                "  g()",
                "def g():",
                "  '''This is a good function.",
                "",
                "  It is emphatically not deprecated.'''"))
        .isEmpty();
  }

  @Test
  public void noWarningsInsideDeprecatedFunctions() {
    Truth.assertThat(
        findIssues(
            "def f():",
            "  '''A deprecated function calling another deprecated function -> no warning",
            "",
            "  Deprecated:",
            "    This function is deprecated.",
            "  '''",
            "  g()",
            "",
            "def g():",
            "  '''Another deprecated function",
            "",
            "  Deprecated:",
            "    This function is deprecated.'''"
        )
    ).isEmpty();
  }

  @Test
  public void shadowingWorks() {
    Truth.assertThat(
            findIssues(
                "def f():",
                "  bad = good",
                "  bad()",
                "def good(): pass",
                "def bad():",
                "  '''This is a deprecated function.",
                "",
                "  Deprecated:",
                "    reason'''"))
        .isEmpty();
  }
}
