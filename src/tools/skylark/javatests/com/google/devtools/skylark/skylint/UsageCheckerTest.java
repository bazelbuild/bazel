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
    Truth.assertThat(message).contains("1:13-1:15: unused binding of 'x' [unused-binding]");
    Truth.assertThat(message).contains("1:18-1:20: unused binding of 'y' [unused-binding]");
    Truth.assertThat(message).contains("1:23-1:24: unused binding of '_z' [unused-binding]");
  }

  @Test
  public void reportUnusedGlobals() throws Exception {
    String message = findIssues("_UNUSED = len([])", "def _unused(): pass").toString();
    Truth.assertThat(message).contains("1:1-1:7: unused binding of '_UNUSED' [unused-binding]");
    Truth.assertThat(message).contains("2:5-2:11: unused binding of '_unused' [unused-binding]");
  }

  @Test
  public void reportUnusedLocals() throws Exception {
    String message = findIssues("def some_function(param):", "  local, local2 = 1, 3").toString();
    Truth.assertThat(message).contains("1:19-1:23: unused binding of 'param'");
    Truth.assertThat(message)
        .contains(
            "you can add `_ignore = [<param1>, <param2>, ...]` to the function body."
                + " [unused-binding]");
    Truth.assertThat(message).contains("2:3-2:7: unused binding of 'local'");
    Truth.assertThat(message)
        .contains("you can use '_' or rename it to '_local'. [unused-binding]");
    Truth.assertThat(message).contains("2:10-2:15: unused binding of 'local2'");
    Truth.assertThat(message)
        .contains("you can use '_' or rename it to '_local2'. [unused-binding]");
  }

  @Test
  public void reportUnusedComprehensionVariable() throws Exception {
    String message = findIssues("[[2 for y in []] for x in []]").toString();
    Truth.assertThat(message).contains("1:9-1:9: unused binding of 'y'");
    Truth.assertThat(message).contains("you can use '_' or rename it to '_y'. [unused-binding]");
    Truth.assertThat(message).contains("1:22-1:22: unused binding of 'x'");
    Truth.assertThat(message).contains("you can use '_' or rename it to '_x'. [unused-binding]");
  }

  @Test
  public void reportShadowingVariable() throws Exception {
    Truth.assertThat(
            findIssues(
                    "def some_function_name_foo_bar_baz_qux():",
                    "  x = [[] for x in []]",
                    "  print(x)")
                .toString())
        .contains("2:15-2:15: unused binding of 'x'");
  }

  @Test
  public void reportShadowedVariable() throws Exception {
    Truth.assertThat(findIssues("def some_function():", "  x = [x for x in []]").toString())
        .contains("2:3-2:3: unused binding of 'x'");
  }

  @Test
  public void reportReassignedUnusedVariable() throws Exception {
    Truth.assertThat(findIssues("def some_function():", "  x = 1", "  x += 2").toString())
        .contains("3:3-3:3: unused binding of 'x'");
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
        .contains("2:3-2:3: unused binding of 'x'");
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
    Truth.assertThat(messages).contains("4:5-4:5: unused binding of 'x'");
  }

  @Test
  public void reportUninitializedAfterIfElifElse() throws Exception {
    String message =
        findIssues(
                "def some_function(a, b):",
                "  if a:",
                "    y = 2",
                "  elif b:",
                "    pass",
                "  else:",
                "    y = 1",
                "  y += 2",
                "  print(y)")
            .toString();
    Truth.assertThat(message)
        .contains("8:3-8:3: variable 'y' may not have been initialized [uninitialized-variable]");
  }

  @Test
  public void reportUninitializedAfterForLoop() throws Exception {
    String message =
        findIssues("def some_function():", "  for _ in []:", "    y = 1", "  print(y)").toString();
    Truth.assertThat(message)
        .contains("4:9-4:9: variable 'y' may not have been initialized [uninitialized-variable]");
  }

  @Test
  public void dontReportAlwaysInitializedInNestedIf() throws Exception {
    Truth.assertThat(
            findIssues(
                "def some_function(a, b):",
                "  if a:",
                "    if b:",
                "      x = b",
                "    else:",
                "      x = a",
                "  else:",
                "    x = not a",
                "  return x"))
        .isEmpty();
  }

  @Test
  public void dontReportAlwaysInitializedBecauseUnreachable() throws Exception {
    Truth.assertThat(
            findIssues(
                "def some_function(a, b):",
                "  if a:",
                "    y = 1",
                "  elif b:",
                "    return",
                "  else:",
                "    fail('fail')",
                "  print(y)",
                "  for _ in []:",
                "    if a:",
                "      break",
                "    elif b:",
                "      continue",
                "    else:",
                "      z = 2",
                "    print(z)"))
        .isEmpty();
  }

  @Test
  public void dontReportUsedAsParameterDefault() throws Exception {
    Truth.assertThat(
        findIssues(
            "_x = 1",
            "def foo(y=_x):",
            "  print(y)",
            "",
            "foo()"
        )
    ).isEmpty();
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
                "    x += 1",
                "  return x",
                "",
                "def foo():",
                "    x = \"xyz\"",
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
                "def public_function():",
                "  _global_function(_GLOBAL)",
                "def _global_function(param):",
                "  print(param)"))
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

  @Test
  public void dontReportInitializationWithNoneAsDeclaration() throws Exception {
    Truth.assertThat(
            findIssues(
                "def foo(bar):",
                "  baz = None # here should be no unused warning",
                "  # because we want to allow people to 'declare' a variable in one location",
                "  if bar:",
                "    baz = 0",
                "  else:",
                "    baz = 1",
                "  print(baz)"))
        .isEmpty();
  }

  @Test
  public void reportUnusedInitializationWithNone() throws Exception {
    Truth.assertThat(
            findIssues("def foo():", "  baz = None # warn here because 'baz' is never used")
                .toString())
        .contains("2:3-2:5: unused binding of 'baz'");
  }

  @Test
  public void reportSubsequentInitializations() throws Exception {
    Truth.assertThat(
            findIssues(
                    "def foo():",
                    "  baz = None",
                    "  baz = None # do warn here (not an initialization)")
                .toString())
        .contains("3:3-3:5: unused binding of 'baz'");
    Truth.assertThat(
            findIssues(
                    "def foo():",
                    "  baz = None",
                    "  baz = 0 # do warn here (it's a regular assignment)")
                .toString())
        .contains("3:3-3:5: unused binding of 'baz'");
    Truth.assertThat(
            findIssues(
                    "def foo():",
                    "  baz = 0",
                    "  baz = None # do warn here (not an initialization)")
                .toString())
        .contains("3:3-3:5: unused binding of 'baz'");
  }
}
