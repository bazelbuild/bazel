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

import com.google.devtools.build.lib.bazel.rules.ninja.file.FileFragment;
import com.google.devtools.build.lib.bazel.rules.ninja.lexer.NinjaLexer;
import com.google.devtools.build.lib.bazel.rules.ninja.lexer.NinjaLexer.TextKind;
import com.google.devtools.build.lib.bazel.rules.ninja.lexer.NinjaToken;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link com.google.devtools.build.lib.bazel.rules.ninja.lexer.NinjaLexer}. */
@RunWith(JUnit4.class)
public class NinjaLexerTest {

  @Test
  public void testReadIdentifiersAndVariables() {
    String text = "abc efg    $fg ${ghf}\ntext one ${ more.1-d_f } $abc.def";
    NinjaLexer lexer = createLexer(text);
    assertTokenBytes(lexer, NinjaToken.IDENTIFIER, "abc");
    assertTokenBytes(lexer, NinjaToken.IDENTIFIER, "efg");
    assertTokenBytes(lexer, NinjaToken.VARIABLE, "$fg");
    assertTokenBytes(lexer, NinjaToken.VARIABLE, "${ghf}");
    assertTokenBytes(lexer, NinjaToken.NEWLINE, null);
    assertTokenBytes(lexer, NinjaToken.IDENTIFIER, "text");
    assertTokenBytes(lexer, NinjaToken.IDENTIFIER, "one");
    assertTokenBytes(lexer, NinjaToken.VARIABLE, "${ more.1-d_f }");
    assertTokenBytes(lexer, NinjaToken.VARIABLE, "$abc");
    assertTokenBytes(lexer, NinjaToken.IDENTIFIER, ".def");
  }

  @Test
  public void testNewlines() {
    String text = "a\nb $\nnot-newline$$\nnewline\n\nand\r\none";
    NinjaLexer lexer = createLexer(text);
    assertTokenBytes(lexer, NinjaToken.IDENTIFIER, "a");
    assertTokenBytes(lexer, NinjaToken.NEWLINE, null);
    assertTokenBytes(lexer, NinjaToken.IDENTIFIER, "b");
    assertTokenBytes(lexer, NinjaToken.IDENTIFIER, "not-newline");
    assertTokenBytes(lexer, NinjaToken.ESCAPED_TEXT, "$$");
    assertTokenBytes(lexer, NinjaToken.NEWLINE, null);
    assertTokenBytes(lexer, NinjaToken.IDENTIFIER, "newline");
    assertTokenBytes(lexer, NinjaToken.NEWLINE, null);
    assertTokenBytes(lexer, NinjaToken.NEWLINE, null);
    assertTokenBytes(lexer, NinjaToken.IDENTIFIER, "and");
    assertTokenBytes(lexer, NinjaToken.NEWLINE, null);
    assertTokenBytes(lexer, NinjaToken.IDENTIFIER, "one");
    assertThat(lexer.hasNextToken()).isFalse();
  }

  @Test
  public void testTabsAreAllowed() {
    String text = "abc\n\tcde";
    NinjaLexer lexer = createLexer(text);
    assertTokenBytes(lexer, NinjaToken.IDENTIFIER, "abc");
    assertTokenBytes(lexer, NinjaToken.NEWLINE, null);
    assertTokenBytes(lexer, NinjaToken.INDENT, null);
    assertTokenBytes(lexer, NinjaToken.IDENTIFIER, "cde");
  }

  @Test
  public void testDisallowedSymbols() {
    assertError(
        createLexer("^"),
        "Symbol '^' is not allowed in the identifier, the text fragment with the symbol:\n^\n",
        "^");
  }

  @Test
  public void testComments() {
    String text =
        "abc#immediately after\n#Start of the line $ not escaped in comment $"
            + "\nNot-comment# Finishing : = $ | ||";
    NinjaLexer lexer = createLexer(text);
    assertTokenBytes(lexer, NinjaToken.IDENTIFIER, "abc");
    assertTokenBytes(lexer, NinjaToken.NEWLINE, null);
    assertTokenBytes(lexer, NinjaToken.NEWLINE, null);
    assertTokenBytes(lexer, NinjaToken.IDENTIFIER, "Not-comment");
  }

  @Test
  public void testBadEscape() {
    NinjaLexer lexer = createLexer("abc\nbad $");
    assertTokenBytes(lexer, NinjaToken.IDENTIFIER, "abc");
    assertTokenBytes(lexer, NinjaToken.NEWLINE, null);
    assertTokenBytes(lexer, NinjaToken.IDENTIFIER, "bad");
    assertError(lexer, "Bad $-escape (literal $ must be written as $$)", "$");

    NinjaLexer lexer2 = createLexer("$$$");
    assertTokenBytes(lexer2, NinjaToken.ESCAPED_TEXT, "$$");
    assertError(lexer2, "Bad $-escape (literal $ must be written as $$)", "$");
  }

  @Test
  public void testBadVariable() {
    assertError(createLexer("${abc"), "Variable end symbol '}' expected.", "${abc");
    assertError(createLexer("${abc "), "Variable end symbol '}' expected.", "${abc ");
    assertError(createLexer("${}"), "Variable identifier expected.", "${}");
    assertError(createLexer("${abc&}"), "Variable end symbol '}' expected.", "${abc&");
  }

  @Test
  public void testKeywords() {
    assertTokenBytes(createLexer("build"), NinjaToken.BUILD, null);
    assertTokenBytes(createLexer("rule"), NinjaToken.RULE, null);
    assertTokenBytes(createLexer("default "), NinjaToken.DEFAULT, null);
    assertTokenBytes(createLexer("include"), NinjaToken.INCLUDE, null);
    assertTokenBytes(createLexer("subninja\n"), NinjaToken.SUBNINJA, null);
    assertTokenBytes(createLexer("pool "), NinjaToken.POOL, null);
  }

  @Test
  public void testIndent() {
    NinjaLexer lexer = createLexer(" a\nb\n  c   d   e\n ");
    // We want to know if there was a starting INDENT
    // (though we suppose to start with a line without INDENT)
    assertTokenBytes(lexer, NinjaToken.INDENT, null);
    assertTokenBytes(lexer, NinjaToken.IDENTIFIER, "a");
    assertTokenBytes(lexer, NinjaToken.NEWLINE, null);
    assertTokenBytes(lexer, NinjaToken.IDENTIFIER, "b");
    assertTokenBytes(lexer, NinjaToken.NEWLINE, null);
    assertTokenBytes(lexer, NinjaToken.INDENT, null);
    assertTokenBytes(lexer, NinjaToken.IDENTIFIER, "c");
    assertTokenBytes(lexer, NinjaToken.IDENTIFIER, "d");
    assertTokenBytes(lexer, NinjaToken.IDENTIFIER, "e");
    assertTokenBytes(lexer, NinjaToken.NEWLINE, null);
    assertTokenBytes(lexer, NinjaToken.INDENT, null);
  }

  @Test
  public void testReadTextFragment() {
    NinjaLexer lexer = createLexer("my.var=Any text ^&%=@&!*: $:symbols$\n aa\nmy.var2");
    assertTokenBytes(lexer, NinjaToken.IDENTIFIER, "my.var");
    assertTokenBytes(lexer, NinjaToken.EQUALS, null);

    lexer.setExpectedTextKind(TextKind.TEXT);
    assertTokenBytes(lexer, NinjaToken.TEXT, "Any");
    assertTokenBytes(lexer, NinjaToken.TEXT, "text");
    assertTokenBytes(lexer, NinjaToken.TEXT, "^&%=@&!*");
    assertTokenBytes(lexer, NinjaToken.COLON, null);
    assertTokenBytes(lexer, NinjaToken.ESCAPED_TEXT, "$:");
    assertTokenBytes(lexer, NinjaToken.TEXT, "symbols");
    assertTokenBytes(lexer, NinjaToken.TEXT, "aa");
    assertTokenBytes(lexer, NinjaToken.NEWLINE, null);
    assertTokenBytes(lexer, NinjaToken.IDENTIFIER, "my.var2");
  }

  @Test
  public void testUndo() {
    NinjaLexer lexer = createLexer("my.var=Any\n");
    assertTokenBytes(lexer, NinjaToken.IDENTIFIER, "my.var");
    assertTokenBytes(lexer, NinjaToken.EQUALS, null);
    assertTokenBytes(lexer, NinjaToken.IDENTIFIER, "Any");
    assertTokenBytes(lexer, NinjaToken.NEWLINE, null);

    lexer.undo();
    assertTokenBytes(lexer, NinjaToken.NEWLINE, null);
    lexer.undo();
    lexer.undo();
    assertTokenBytes(lexer, NinjaToken.IDENTIFIER, "Any");
    assertTokenBytes(lexer, NinjaToken.NEWLINE, null);
    lexer.undo();
    lexer.undo();
    lexer.undo();
    assertTokenBytes(lexer, NinjaToken.EQUALS, null);
    assertTokenBytes(lexer, NinjaToken.IDENTIFIER, "Any");
    assertTokenBytes(lexer, NinjaToken.NEWLINE, null);
    lexer.undo();
    lexer.undo();
    lexer.undo();
    lexer.undo();
    assertTokenBytes(lexer, NinjaToken.IDENTIFIER, "my.var");
    lexer.undo();
    assertTokenBytes(lexer, NinjaToken.IDENTIFIER, "my.var");
    assertTokenBytes(lexer, NinjaToken.EQUALS, null);
    assertTokenBytes(lexer, NinjaToken.IDENTIFIER, "Any");
    assertTokenBytes(lexer, NinjaToken.NEWLINE, null);
  }

  @Test
  public void testSpecialSymbols() {
    NinjaLexer lexer = createLexer("| || : = ");
    assertTokenBytes(lexer, NinjaToken.PIPE, null);
    assertTokenBytes(lexer, NinjaToken.PIPE2, null);
    assertTokenBytes(lexer, NinjaToken.COLON, null);
    assertTokenBytes(lexer, NinjaToken.EQUALS, null);
    assertTokenBytes(lexer, NinjaToken.EOF, null);
    assertThat(lexer.hasNextToken()).isFalse();
  }

  @Test
  public void testZeroByte() {
    byte[] bytes = {'a', 0, 'b'};
    NinjaLexer lexer = new NinjaLexer(new FileFragment(ByteBuffer.wrap(bytes), 0, 0, bytes.length));
    assertTokenBytes(lexer, NinjaToken.IDENTIFIER, null);
    assertThat(lexer.hasNextToken()).isFalse();
  }

  private static void assertError(NinjaLexer lexer, String errorText, String errorHolder) {
    assertThat(lexer.hasNextToken()).isTrue();
    assertThat(lexer.nextToken()).isEqualTo(NinjaToken.ERROR);
    assertThat(lexer.getError()).isEqualTo(errorText);
    assertThat(lexer.getTokenBytes()).isEqualTo(errorHolder.getBytes(StandardCharsets.ISO_8859_1));
    assertThat(lexer.hasNextToken()).isFalse();
  }

  private static void assertTokenBytes(NinjaLexer lexer, NinjaToken token, @Nullable String text) {
    assertThat(lexer.hasNextToken()).isTrue();
    assertThat(lexer.nextToken()).isEqualTo(token);
    if (text != null) {
      assertThat(lexer.getTokenBytes()).isEqualTo(text.getBytes(StandardCharsets.ISO_8859_1));
    }
  }

  private static NinjaLexer createLexer(String text) {
    ByteBuffer buffer = ByteBuffer.wrap(text.getBytes(StandardCharsets.ISO_8859_1));
    return new NinjaLexer(new FileFragment(buffer, 0, 0, buffer.limit()));
  }
}
