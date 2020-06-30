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
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests of StarlarkFile parsing. */
// TODO(adonovan): move tests of parsing into ParserTest.
@RunWith(JUnit4.class)
public class StarlarkFileTest {

  /**
   * Parses the contents of the specified string (using 'foo.star' as the apparent filename) and
   * returns the AST. Resets the error handler beforehand.
   */
  private static StarlarkFile parseFile(String... lines) {
    String src = Joiner.on("\n").join(lines);
    ParserInput input = ParserInput.fromString(src, "foo.star");
    return StarlarkFile.parse(input);
  }

  @Test
  public void testParsesFineWithNewlines() throws Exception {
    StarlarkFile file = parseFile("foo()", "bar()", "something = baz()", "bar()");
    assertThat(file.getStatements()).hasSize(4);
  }

  @Test
  public void testFailsIfNewlinesAreMissing() throws Exception {
    StarlarkFile file = parseFile("foo() bar() something = baz() bar()");

    SyntaxError error =
        LexerTest.assertContainsError(file.errors(), "syntax error at \'bar\': expected newline");
    assertThat(error.location().toString()).isEqualTo("foo.star:1:7");
  }

  @Test
  public void testImplicitStringConcatenationFails() throws Exception {
    // TODO(adonovan): move to ParserTest.
    StarlarkFile file = parseFile("a = 'foo' 'bar'");
    SyntaxError error =
        LexerTest.assertContainsError(
            file.errors(), "Implicit string concatenation is forbidden, use the + operator");
    assertThat(error.location().toString()).isEqualTo("foo.star:1:11"); // start of 'bar'
  }

  @Test
  public void testImplicitStringConcatenationAcrossLinesIsIllegal() throws Exception {
    StarlarkFile file = parseFile("a = 'foo'\n  'bar'");

    SyntaxError error = LexerTest.assertContainsError(file.errors(), "indentation error");
    assertThat(error.location().toString()).isEqualTo("foo.star:2:2");
  }
}
