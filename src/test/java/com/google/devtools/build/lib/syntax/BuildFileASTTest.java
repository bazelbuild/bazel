// Copyright 2006 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.syntax;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.syntax.SkylarkList.Tuple;
import com.google.devtools.build.lib.syntax.util.EvaluationTestCase;
import com.google.devtools.build.lib.testutil.MoreAsserts;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for BuildFileAST.
 */
@RunWith(JUnit4.class)
public class BuildFileASTTest extends EvaluationTestCase {

  private Scratch scratch = new Scratch();

  @Override
  public Environment newEnvironment() throws Exception {
    return newBuildEnvironment();
  }

  /**
   * Parses the contents of the specified string (using DUMMY_PATH as the fake
   * filename) and returns the AST. Resets the error handler beforehand.
   */
  private BuildFileAST parseBuildFile(String... lines) throws IOException {
    Path file = scratch.file("/a/build/file/BUILD", lines);
    return BuildFileAST.parseBuildFile(file, file.getFileSize(), getEventHandler());
  }

  @Test
  public void testParseBuildFileOK() throws Exception {
    BuildFileAST buildfile = parseBuildFile(
        "# a file in the build language",
        "",
        "x = [1,2,'foo',4] + [1,2, \"%s%d\" % ('foo', 1)]");

    assertTrue(buildfile.exec(env, getEventHandler()));

    // Test final environment is correctly modified:
    //
    // input1.BUILD contains:
    // x = [1,2,'foo',4] + [1,2, "%s%d" % ('foo', 1)]
    assertEquals(SkylarkList.createImmutable(Tuple.of(1, 2, "foo", 4, 1, 2, "foo1")),
        env.lookup("x"));
  }

  @Test
  public void testEvalException() throws Exception {
    setFailFast(false);
    BuildFileAST buildfile = parseBuildFile(
        "x = 1",
        "y = [2,3]",
        "",
        "z = x + y");

    assertFalse(buildfile.exec(env, getEventHandler()));
    Event e = assertContainsError("unsupported operand type(s) for +: 'int' and 'list'");
    assertEquals(4, e.getLocation().getStartLineAndColumn().getLine());
  }

  @Test
  public void testParsesFineWithNewlines() throws Exception {
    BuildFileAST buildFileAST = parseBuildFile(
        "foo()",
        "bar(),",
        "something = baz()",
        "bar()");
    assertThat(buildFileAST.getStatements()).hasSize(4);
  }

  @Test
  public void testFailsIfNewlinesAreMissing() throws Exception {
    setFailFast(false);

    BuildFileAST buildFileAST =
      parseBuildFile("foo() bar() something = baz() bar()");

    Event event = assertContainsError("syntax error at \'bar\': expected newline");
    assertEquals("/a/build/file/BUILD",
                 event.getLocation().getPath().toString());
    assertEquals(1, event.getLocation().getStartLineAndColumn().getLine());
    assertTrue(buildFileAST.containsErrors());
  }

  @Test
  public void testImplicitStringConcatenationFails() throws Exception {
    setFailFast(false);
    BuildFileAST buildFileAST = parseBuildFile("a = 'foo' 'bar'");
    Event event = assertContainsError(
        "Implicit string concatenation is forbidden, use the + operator");
    assertEquals("/a/build/file/BUILD",
                 event.getLocation().getPath().toString());
    assertEquals(1, event.getLocation().getStartLineAndColumn().getLine());
    assertEquals(10, event.getLocation().getStartLineAndColumn().getColumn());
    assertTrue(buildFileAST.containsErrors());
  }

  @Test
  public void testImplicitStringConcatenationAcrossLinesIsIllegal() throws Exception {
    setFailFast(false);
    BuildFileAST buildFileAST = parseBuildFile("a = 'foo'\n  'bar'");

    Event event = assertContainsError("indentation error");
    assertEquals("/a/build/file/BUILD",
                 event.getLocation().getPath().toString());
    assertEquals(2, event.getLocation().getStartLineAndColumn().getLine());
    assertEquals(2, event.getLocation().getStartLineAndColumn().getColumn());
    assertTrue(buildFileAST.containsErrors());
  }

  @Test
  public void testWithSyntaxErrorsDoesNotPrintDollarError() throws Exception {
    setFailFast(false);
    BuildFileAST buildFile = parseBuildFile(
        "abi = cxx_abi + '-glibc-' + glibc_version + '-' + $(TARGET_CPU) + '-linux'",
        "libs = [abi + opt_level + '/lib/libcc.a']",
        "shlibs = [abi + opt_level + '/lib/libcc.so']",
        "+* shlibs", // syntax error at '+'
        "cc_library(name = 'cc',",
        "           srcs = libs,",
        "           includes = [ abi + opt_level + '/include' ])");
    assertTrue(buildFile.containsErrors());
    assertContainsError("syntax error at '+': expected expression");
    assertFalse(buildFile.exec(env, getEventHandler()));
    MoreAsserts.assertDoesNotContainEvent(getEventCollector(), "$error$");
    // This message should not be printed anymore.
    MoreAsserts.assertDoesNotContainEvent(getEventCollector(), "contains syntax error(s)");
  }
}
