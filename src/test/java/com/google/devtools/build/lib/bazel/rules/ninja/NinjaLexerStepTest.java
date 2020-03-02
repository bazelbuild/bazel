// Copyright 2019 The Bazel Authors. All rights reserved.
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
//

package com.google.devtools.build.lib.bazel.rules.ninja;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.bazel.rules.ninja.file.ByteBufferFragment;
import com.google.devtools.build.lib.bazel.rules.ninja.lexer.NinjaLexerStep;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.function.Consumer;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link com.google.devtools.build.lib.bazel.rules.ninja.lexer.NinjaLexerStep}. */
@RunWith(JUnit4.class)
public class NinjaLexerStepTest {
  @Test
  public void testSkipSpaces() {
    NinjaLexerStep step = step("  ");
    assertThat(step.canAdvance()).isTrue();
    assertThat(step.startByte()).isEqualTo(' ');
    step.skipSpaces();
    assertThat(step.getEnd()).isEqualTo(2);
    assertThat(step.getBytes()).isEqualTo("  ".getBytes(StandardCharsets.ISO_8859_1));
    assertThat(step.canAdvance()).isFalse();
  }

  @Test
  public void testSkipComment() {
    doTest("# skfdskl werj wior982438923u rekfjg wef w $ $: : |", NinjaLexerStep::skipComment);
    doTest("#", NinjaLexerStep::skipComment);
    doTest("# 123\n", NinjaLexerStep::skipComment, "# 123", true);
    doTest("# 123\u0000118", NinjaLexerStep::skipComment, "# 123", false);
  }

  @Test
  public void testTrySkipEscapedNewline() {
    doTest("$\n", step -> assertThat(step.trySkipEscapedNewline()).isTrue());
    doTest("$\n\n", step -> assertThat(step.trySkipEscapedNewline()).isTrue(), "$\n", true);
    doTest("$\r\n", step -> assertThat(step.trySkipEscapedNewline()).isTrue());
    doTest("$\r\na", step -> assertThat(step.trySkipEscapedNewline()).isTrue(), "$\r\n", true);
  }

  @Test
  public void testProcessLineFeedNewLine() {
    doTest("\r\n", NinjaLexerStep::processLineFeedNewLine);
    doTestError(
        "\ra",
        NinjaLexerStep::processLineFeedNewLine,
        "\ra",
        false,
        "Wrong newline separators: \\r should be followed by \\n.");
  }

  @Test
  public void testTryReadVariableInBrackets() {
    doTest("${abc}", step -> assertThat(step.tryReadVariableInBrackets()).isTrue());
    doTest("$abc", step -> assertThat(step.tryReadVariableInBrackets()).isFalse(), "", true);

    doTest("${  abc  }", step -> assertThat(step.tryReadVariableInBrackets()).isTrue());
    doTest(
        "${abc.xyz-1_2}cde",
        step -> assertThat(step.tryReadVariableInBrackets()).isTrue(),
        "${abc.xyz-1_2}",
        true);
    doTestError(
        "${abc",
        step -> assertThat(step.tryReadVariableInBrackets()).isTrue(),
        "${abc",
        false,
        "Variable end symbol '}' expected.");
    doTestError(
        "${abc\n",
        step -> assertThat(step.tryReadVariableInBrackets()).isTrue(),
        "${abc\n",
        false,
        "Variable end symbol '}' expected.");
    doTestError(
        "${}",
        step -> assertThat(step.tryReadVariableInBrackets()).isTrue(),
        "${}",
        false,
        "Variable identifier expected.");
    doTestError(
        "${^}",
        step -> assertThat(step.tryReadVariableInBrackets()).isTrue(),
        "${^",
        true,
        "Variable identifier expected.");
  }

  @Test
  public void testTryReadSimpleVariable() {
    doTest("$abc", step -> assertThat(step.tryReadSimpleVariable()).isTrue());
    doTest("$a-b_c", step -> assertThat(step.tryReadSimpleVariable()).isTrue());
    doTest("$.", step -> assertThat(step.tryReadSimpleVariable()).isFalse(), "", true);
    doTest("$abc.cde", step -> assertThat(step.tryReadSimpleVariable()).isTrue(), "$abc", true);
  }

  @Test
  public void testTryReadEscapedLiteral() {
    doTest("$:", step -> assertThat(step.tryReadEscapedLiteral()).isTrue(), "$:", false);
    doTest("$$", step -> assertThat(step.tryReadEscapedLiteral()).isTrue(), "$$", false);
    doTest("$ ", step -> assertThat(step.tryReadEscapedLiteral()).isTrue(), "$ ", false);

    doTest("$:a", step -> assertThat(step.tryReadEscapedLiteral()).isTrue(), "$:", true);
    doTest("$$$", step -> assertThat(step.tryReadEscapedLiteral()).isTrue(), "$$", true);
    doTest("$  ", step -> assertThat(step.tryReadEscapedLiteral()).isTrue(), "$ ", true);

    doTest("$a", step -> assertThat(step.tryReadEscapedLiteral()).isFalse(), "", true);
  }

  @Test
  public void testTryReadIdentifier() {
    doTest("abc_d-18", NinjaLexerStep::tryReadIdentifier);
    doTest("abc_d-18.ccc", NinjaLexerStep::tryReadIdentifier);
    doTest("abc_d-18.ccc=", NinjaLexerStep::tryReadIdentifier, "abc_d-18.ccc", true);
    // Have a longer text to demonstrate the error output.
    doTestError(
        "^abc Bazel only rebuilds what is necessary. "
            + "With advanced local and distributed caching, optimized dependency analysis "
            + "and parallel execution, you get fast and incremental builds.",
        NinjaLexerStep::tryReadIdentifier,
        "^",
        true,
        "Symbol '^' is not allowed in the identifier, the text fragment with the symbol:\n"
            + "^abc Bazel only rebuilds what is necessary. With advanced local and distributed"
            + " caching, optimized dependency analysis and parallel execution,"
            + " you get fast and incremental builds.\n");
  }

  @Test
  public void testReadPath() {
    doTest(
        "this/is/the/relative/path.txt",
        NinjaLexerStep::readPath,
        "this/is/the/relative/path.txt",
        false);
    doTest(
        "relative/text#.properties", NinjaLexerStep::readPath, "relative/text#.properties", false);
  }

  @Test
  public void testTryReadDoublePipe() {
    doTest("||", NinjaLexerStep::tryReadDoublePipe);
  }

  @Test
  public void testReadText() {
    doTest("text$\npart", NinjaLexerStep::readText, "text", true);
    doTest("one word", NinjaLexerStep::readText, "one", true);
  }

  private static void doTest(String text, Consumer<NinjaLexerStep> callback) {
    doTest(text, callback, text, false);
  }

  private static void doTest(
      String text, Consumer<NinjaLexerStep> callback, String expected, boolean haveMore) {
    NinjaLexerStep step = step(text);
    assertThat(step.canAdvance()).isTrue();
    callback.accept(step);
    assertThat(step.getError()).isNull();
    if (!expected.isEmpty()) {
      assertThat(step.getBytes()).isEqualTo(expected.getBytes(StandardCharsets.ISO_8859_1));
    }
    assertThat(step.canAdvance()).isEqualTo(haveMore);
  }

  private static void doTestError(
      String text,
      Consumer<NinjaLexerStep> callback,
      String expected,
      boolean haveMore,
      String errorText) {
    NinjaLexerStep step = step(text);
    assertThat(step.canAdvance()).isTrue();
    callback.accept(step);
    assertThat(step.getError()).isEqualTo(errorText);
    assertThat(step.getBytes()).isEqualTo(expected.getBytes(StandardCharsets.ISO_8859_1));
    assertThat(step.getFragment().length() > step.getEnd()).isEqualTo(haveMore);
  }

  private static NinjaLexerStep step(String text) {
    ByteBuffer bb = ByteBuffer.wrap(text.getBytes(StandardCharsets.ISO_8859_1));
    return new NinjaLexerStep(new ByteBufferFragment(bb, 0, bb.limit()), 0);
  }
}
