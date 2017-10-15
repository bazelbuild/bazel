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

@RunWith(JUnit4.class)
public class ControlFlowCheckerTest {
  private static List<Issue> findIssues(String... lines) {
    String content = String.join("\n", lines);
    BuildFileAST ast =
        BuildFileAST.parseString(
            event -> {
              throw new IllegalArgumentException(event.getMessage());
            },
            content);
    return ControlFlowChecker.check(ast);
  }

  @Test
  public void testAnalyzerToleratesTopLevelFail() throws Exception {
    Truth.assertThat(
            findIssues("fail(\"fail is considered a return, but not at the top level\")"))
        .isEmpty();
  }

  @Test
  public void testIfElseReturnMissing() throws Exception {
    Truth.assertThat(
            findIssues(
                    "def some_function(x):",
                    "  if x:",
                    "    print('foo')",
                    "  else:",
                    "    return x")
                .toString())
        .contains("some but not all execution paths of 'some_function' return a value");
  }

  @Test
  public void testIfElseReturnValueMissing() throws Exception {
    String messages =
        findIssues(
                "def some_function(x):",
                "  if x:",
                "    return x",
                "  else:",
                "    return # missing value")
            .toString();
    Truth.assertThat(messages)
        .contains("some but not all execution paths of 'some_function' return a value");
    Truth.assertThat(messages).contains(":5:5: return value missing");
  }

  @Test
  public void testIfElifElseReturnMissing() throws Exception {
    Truth.assertThat(
            findIssues(
                    "def f(x):",
                    "  if x:",
                    "    return x",
                    "  elif not x:",
                    "    pass",
                    "  else:",
                    "    return not x")
                .toString())
        .contains("some but not all execution paths of 'f' return a value");
  }

  @Test
  public void testNestedIfElseReturnMissing() throws Exception {
    Truth.assertThat(
            findIssues(
                    "def f(x, y):",
                    "  if x:",
                    "    if y:",
                    "      return y",
                    "    else:",
                    "      print('foo')",
                    "  else:",
                    "    return x")
                .toString())
        .contains("some but not all execution paths of 'f' return a value");
  }

  @Test
  public void testElseBranchMissing() throws Exception {
    Truth.assertThat(
            findIssues(
                    "def some_function(x):",
                    "  if x:",
                    "    return x",
                    "  elif not x:",
                    "    return not x")
                .toString())
        .contains("some but not all execution paths of 'some_function' return a value");
  }

  @Test
  public void testIfAndFallOffEnd() throws Exception {
    Truth.assertThat(
            findIssues(
                    "def some_function(x):",
                    "  if x:",
                    "    return x",
                    "  print('foo')",
                    "  # return missing here")
                .toString())
        .contains("some but not all execution paths of 'some_function' return a value");
  }

  @Test
  public void testForAndFallOffEnd() throws Exception {
    Truth.assertThat(
            findIssues(
                    "def some_function():",
                    "  for x in []:",
                    "    return x",
                    "  print('foo')",
                    "  # return missing here")
                .toString())
        .contains("some but not all execution paths of 'some_function' return a value");
  }

  @Test
  public void testAlwaysReturnButSometimesWithoutValue() throws Exception {
    String messages =
        findIssues(
                "def some_function(x):",
                "  if x:",
                "    return # returns without value here",
                "  return x")
            .toString();
    Truth.assertThat(messages)
        .contains("some but not all execution paths of 'some_function' return a value");
    Truth.assertThat(messages).contains(":3:5: return value missing");
  }

  @Test
  public void testUnreachableAfterIf() throws Exception {
    String messages =
        findIssues(
                "def some_function(x):",
                "  if x:",
                "    return",
                "  else:",
                "    fail('fail')",
                "  print('This line is unreachable')")
            .toString();
    Truth.assertThat(messages).contains(":6:3: unreachable statement");
  }

  @Test
  public void testNoUnreachableDuplicates() throws Exception {
    List<Issue> messages =
        findIssues(
            "def some_function():",
            "  return",
            "  print('unreachable1')",
            "  print('unreachable2')");
    Truth.assertThat(messages).hasSize(1);
  }

  @Test
  public void testUnreachableAfterBreakContinue() throws Exception {
    String messages =
        findIssues(
                "def some_function(x):",
                "  for y in x:",
                "    if y:",
                "      break",
                "    else:",
                "      continue",
                "    print('unreachable')")
            .toString();
    Truth.assertThat(messages).contains(":7:5: unreachable statement");
  }

  @Test
  public void testReachableStatements() throws Exception {
    Truth.assertThat(
            findIssues(
                "def some_function(x):",
                "  if x:",
                "    return",
                "  for y in []:",
                "    if y:",
                "      continue",
                "    else:",
                "      fail('fail')",
                "  return"))
        .isEmpty();
  }

  @Test
  public void testIfBeforeReturn() throws Exception {
    Truth.assertThat(
            findIssues(
                "def f(x, y):",
                "  if x:",
                "    return x",
                "  elif not y:",
                "    print('foo')",
                "  print('bar')",
                "  return y"))
        .isEmpty();
  }

  @Test
  public void testReturnInAllBranches() throws Exception {
    Truth.assertThat(
            findIssues(
                "def f(x, y):",
                "  if x:",
                "    return x",
                "  elif not y:",
                "    return None",
                "  else:",
                "    return y"))
        .isEmpty();
  }

  @Test
  public void testReturnInNestedIf() throws Exception {
    Truth.assertThat(
            findIssues(
                "def f(x,y):",
                "  if x:",
                "    if y:",
                "      return y",
                "    else:",
                "      return not y",
                "  else:",
                "    return not x"))
        .isEmpty();
  }

  @Test
  public void testIfStatementSequence() throws Exception {
    Truth.assertThat(
            findIssues(
                "def f(x,y):",
                "  if x:",
                "    print('foo')",
                "  else:",
                "    return x",
                "  print('bar')",
                "  if y:",
                "    return x",
                "  else:",
                "    return y"))
        .isEmpty();
    List<Issue> issues =
        findIssues(
            "def f(x,y):",
            "  if x:",
            "    return x",
            "  else:",
            "    return x",
            "  # from now on everything's unreachable",
            "  print('bar')",
            "  if y:",
            "    return x",
            "  # no else branch but doesn't matter since it's unreachable");
    Truth.assertThat(issues).hasSize(1);
    Truth.assertThat(issues.toString()).contains(":7:3: unreachable statement");
  }

  @Test
  public void testCallToFail() throws Exception {
    Truth.assertThat(
            findIssues(
                "def some_function_name(x):",
                "  if x:",
                "    fail('bar')",
                "  else:",
                "    return x"))
        .isEmpty();
  }
}
