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

import com.google.common.base.Joiner;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.testutil.MoreAsserts;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for StarlarkFile. */
// TODO(adonovan): move tests of parser into ParserTest
// and tests of evaluator into Starlark scripts.
@RunWith(JUnit4.class)
public class StarlarkFileTest {

  private static StarlarkThread newThread() {
    return StarlarkThread.builder(Mutability.create("test")).useDefaultSemantics().build();
  }

  /**
   * Parses the contents of the specified string (using 'foo.star' as the apparent filename) and
   * returns the AST. Resets the error handler beforehand.
   */
  private static StarlarkFile parseFile(String... lines) {
    String src = Joiner.on("\n").join(lines);
    ParserInput input = ParserInput.create(src, "foo.star");
    return StarlarkFile.parse(input);
  }

  @Test
  public void testExecuteBuildFileOK() throws Exception {
    StarlarkFile file =
        parseFile(
            "# a file in the build language",
            "",
            "x = [1,2,'foo',4] + [1,2, \"%s%d\" % ('foo', 1)]");
    StarlarkThread thread = newThread();
    Module module = thread.getGlobals();
    EvalUtils.exec(file, module, thread);

    // Test final environment is correctly modified:
    //
    // input1.BUILD contains:
    // x = [1,2,'foo',4] + [1,2, "%s%d" % ('foo', 1)]
    assertThat(thread.getGlobals().lookup("x"))
        .isEqualTo(StarlarkList.of(/*mutability=*/ null, 1, 2, "foo", 4, 1, 2, "foo1"));
  }

  @Test
  public void testExecException() throws Exception {
    StarlarkFile file = parseFile("x = 1", "y = [2,3]", "", "z = x + y");

    StarlarkThread thread = newThread();
    Module module = thread.getGlobals();
    try {
      EvalUtils.exec(file, module, thread);
      throw new AssertionError("execution succeeded unexpectedly");
    } catch (EvalException ex) {
      assertThat(ex.getMessage()).contains("unsupported binary operation: int + list");
      assertThat(ex.getLocation().line()).isEqualTo(4);
    }
  }

  @Test
  public void testParsesFineWithNewlines() throws Exception {
    StarlarkFile file = parseFile("foo()", "bar()", "something = baz()", "bar()");
    assertThat(file.getStatements()).hasSize(4);
  }

  @Test
  public void testFailsIfNewlinesAreMissing() throws Exception {
    StarlarkFile file = parseFile("foo() bar() something = baz() bar()");

    Event event =
        MoreAsserts.assertContainsEvent(file.errors(), "syntax error at \'bar\': expected newline");
    assertThat(event.getLocation().toString()).isEqualTo("foo.star:1:7");
  }

  @Test
  public void testImplicitStringConcatenationFails() throws Exception {
    StarlarkFile file = parseFile("a = 'foo' 'bar'");
    Event event =
        MoreAsserts.assertContainsEvent(
            file.errors(), "Implicit string concatenation is forbidden, use the + operator");
    assertThat(event.getLocation().toString()).isEqualTo("foo.star:1:10");
  }

  @Test
  public void testImplicitStringConcatenationAcrossLinesIsIllegal() throws Exception {
    StarlarkFile file = parseFile("a = 'foo'\n  'bar'");

    Event event = MoreAsserts.assertContainsEvent(file.errors(), "indentation error");
    assertThat(event.getLocation().toString()).isEqualTo("foo.star:2:2");
  }
}
