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

/** Tests the lint done by {@link NamingConventionsChecker}. */
@RunWith(JUnit4.class)
public class UsageCheckerTest {
  private static List<Issue> findIssues(String... lines) {
    String content = String.join("\n", lines);
    BuildFileAST ast =
        BuildFileAST.parseString(
            event -> {
              throw new IllegalArgumentException(event.getMessage());
            },
            content);
    return UsageChecker.check(ast);
  }

  @Test
  public void reportUnusedImports() throws Exception {
    String message = findIssues("load('foo', 'x', 'y', _z = 'Z')").toString();
    Truth.assertThat(message).contains(":1:13: unused definition of 'x'");
    Truth.assertThat(message).contains(":1:18: unused definition of 'y'");
    Truth.assertThat(message).contains(":1:23: unused definition of '_z'");
  }

  @Test
  public void reportUnusedLocals() throws Exception {
    String message = findIssues("def some_function(param):", "  local, local2 = 1, 3").toString();
    Truth.assertThat(message).contains(":1:19: unused definition of 'param'");
    Truth.assertThat(message)
        .contains("you can add `_ignore = [<param1>, <param2>, ...]` to the function body.");
    Truth.assertThat(message).contains(":2:3: unused definition of 'local'");
    Truth.assertThat(message).contains("you can use '_' or rename it to '_local'");
    Truth.assertThat(message).contains(":2:10: unused definition of 'local2'");
    Truth.assertThat(message).contains("you can use '_' or rename it to '_local2'");
  }

  @Test
  public void reportUnusedComprehensionVariable() throws Exception {
    String message = findIssues("[[2 for y in []] for x in []]").toString();
    Truth.assertThat(message).contains(":1:9: unused definition of 'y'");
    Truth.assertThat(message).contains("you can use '_' or rename it to '_y'");
    Truth.assertThat(message).contains(":1:22: unused definition of 'x'");
    Truth.assertThat(message).contains("you can use '_' or rename it to '_x'");
  }

  @Test
  public void reportShadowingVariable() throws Exception {
    Truth.assertThat(
            findIssues(
                    "def some_function_name_foo_bar_baz_qux():",
                    "  x = [[] for x in []]",
                    "  print(x)")
                .toString())
        .contains(":2:15: unused definition of 'x'");
  }

  @Test
  public void reportShadowedVariable() throws Exception {
    Truth.assertThat(findIssues("def some_function():", "  x = [x for x in []]").toString())
        .contains(":2:3: unused definition of 'x'");
  }

  @Test
  public void reportUnusedGlobals() throws Exception {
    String message = findIssues("_UNUSED = len([])", "def _unused(): pass").toString();
    Truth.assertThat(message).contains(":1:1: unused definition of '_UNUSED'");
    Truth.assertThat(message).contains(":2:5: unused definition of '_unused'");
  }

  @Test
  public void reportReassignedUnusedVariable() throws Exception {
    Truth.assertThat(
            findIssues("def some_function():", "  x = 1", "  print(x)", "  x += 2").toString())
        .contains(":4:3: unused definition of 'x'");
  }

  @Test
  public void reportUnusedBeforeIfElse() throws Exception {
    Truth.assertThat(
            findIssues(
                    "def f(y):",
                    "  x = 2",
                    "  if y:",
                    "    x = 3",
                    "  else:",
                    "    x = 4",
                    "  print(x)")
                .toString())
        .contains(":2:3: unused definition of 'x'");
  }

  @Test
  public void reportUnusedInForLoop() throws Exception {
    String messages =
        findIssues(
                "def some_function_foo_bar_baz_qux():",
                "  for x in []:",
                "    print(x)",
                "    x = 2")
            .toString();
    Truth.assertThat(messages).contains(":4:5: unused definition of 'x'");
  }

  @Test
  public void dontReportUsedAfterIf() throws Exception {
    Truth.assertThat(
            findIssues(
                "def some_function(parameter):",
                "  x = 2",
                "  if parameter:",
                "    x = 3",
                "  print(x)"))
        .isEmpty();
  }

  @Test
  public void dontReportUsedInNextIteration() throws Exception {
    Truth.assertThat(
            findIssues(
                "def some_function_foo_bar_baz_qux():",
                "  x = 0",
                "  for _ in []:",
                "    print(x)",
                "    x += 1"))
        .isEmpty();
    Truth.assertThat(
            findIssues(
                "def foo():",
                "    for i in range(5):",
                "        if i % 2 == 1:",
                "            print(x)",
                "        else:",
                "            x = \"abc\""))
        .isEmpty();
  }

  @Test
  public void dontReportUnusedBuiltins() throws Exception {
    Truth.assertThat(findIssues()).isEmpty();
  }

  @Test
  public void dontReportPublicGlobals() throws Exception {
    Truth.assertThat(
            findIssues(
                "GLOBAL = 0",
                "def global_function_name_foo_bar_baz_qux(parameter):",
                "  print(parameter)"))
        .isEmpty();
  }

  @Test
  public void dontReportUsedGlobals() throws Exception {
    Truth.assertThat(
            findIssues(
                "_GLOBAL = 0",
                "def _global_function(param):",
                "  print(param)",
                "_global_function(_GLOBAL)"))
        .isEmpty();
  }

  @Test
  public void dontReportUsedLocals() throws Exception {
    Truth.assertThat(
            findIssues(
                "def f(x,y):",
                "  a = x",
                "  b = x",
                "  if x:",
                "    a = y",
                "  if y:",
                "    b = a",
                "  print(b)"))
        .isEmpty();
  }

  @Test
  public void dontReportUnderscore() throws Exception {
    Truth.assertThat(findIssues("GLOBAL = [1 for _ in []]")).isEmpty();
  }

  @Test
  public void dontReportLocalsStartingWithUnderscore() throws Exception {
    Truth.assertThat(findIssues("def f(_param):", "  _local = [[] for _x in []]")).isEmpty();
  }
}
