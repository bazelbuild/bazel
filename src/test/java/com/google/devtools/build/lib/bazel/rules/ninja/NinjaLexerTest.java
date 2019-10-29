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
import com.google.devtools.build.lib.bazel.rules.ninja.lexer.NinjaLexer;
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
    assertTokenBytes(lexer, NinjaToken.identifier, "abc");
    assertTokenBytes(lexer, NinjaToken.identifier, "efg");
    assertTokenBytes(lexer, NinjaToken.variable, "$fg");
    assertTokenBytes(lexer, NinjaToken.variable, "${ghf}");
    assertTokenBytes(lexer, NinjaToken.newline, null);
    assertTokenBytes(lexer, NinjaToken.identifier, "text");
    assertTokenBytes(lexer, NinjaToken.identifier, "one");
    assertTokenBytes(lexer, NinjaToken.variable, "${ more.1-d_f }");
    assertTokenBytes(lexer, NinjaToken.variable, "$abc");
    assertTokenBytes(lexer, NinjaToken.identifier, ".def");
  }

  @Test
  public void testNewLines() {
    String text = "a\nb $\nnot-newline$$\nnewline\n\nand\r\none";
    NinjaLexer lexer = createLexer(text);
    assertTokenBytes(lexer, NinjaToken.identifier, "a");
    assertTokenBytes(lexer, NinjaToken.newline, null);
    assertTokenBytes(lexer, NinjaToken.identifier, "b");
    assertTokenBytes(lexer, NinjaToken.identifier, "not-newline");
    assertTokenBytes(lexer, NinjaToken.text, "$");
    assertTokenBytes(lexer, NinjaToken.newline, null);
    assertTokenBytes(lexer, NinjaToken.identifier, "newline");
    assertTokenBytes(lexer, NinjaToken.newline, null);
    assertTokenBytes(lexer, NinjaToken.newline, null);
    assertTokenBytes(lexer, NinjaToken.identifier, "and");
    assertTokenBytes(lexer, NinjaToken.newline, null);
    assertTokenBytes(lexer, NinjaToken.identifier, "one");
    assertThat(lexer.hasNextToken()).isFalse();
  }

  @Test
  public void testDisallowedSymbols() {
    String text = "abc\n\tcde";
    NinjaLexer lexer = createLexer(text);
    assertTokenBytes(lexer, NinjaToken.identifier, "abc");
    assertTokenBytes(lexer, NinjaToken.newline, null);
    assertError(lexer, "Tabs are not allowed, use spaces.", "\t");

    assertError(createLexer("^"), "Symbol is not allowed in the identifier.", "^");
  }

  private void assertError(NinjaLexer lexer, String errorText, String errorHolder) {
    assertThat(lexer.hasNextToken()).isTrue();
    assertThat(lexer.nextToken()).isEqualTo(NinjaToken.error);
    assertThat(lexer.getError()).isEqualTo(errorText);
    assertThat(lexer.getTokenBytes()).isEqualTo(errorHolder.getBytes(StandardCharsets.ISO_8859_1));
    assertThat(lexer.hasNextToken()).isFalse();
  }

  @Test
  public void testComments() {
    String text = "abc#immediately after\n#Start of the line $ not escaped in comment $" +
        "\nNot-comment# Finishing : = $ | ||";
    NinjaLexer lexer = createLexer(text);
    assertTokenBytes(lexer, NinjaToken.identifier, "abc");
    assertTokenBytes(lexer, NinjaToken.newline, null);
    assertTokenBytes(lexer, NinjaToken.newline, null);
    assertTokenBytes(lexer, NinjaToken.identifier, "Not-comment");
  }

  @Test
  public void testBadEscape() {
    NinjaLexer lexer = createLexer("abc\nbad $");
    assertTokenBytes(lexer, NinjaToken.identifier, "abc");
    assertTokenBytes(lexer, NinjaToken.newline, null);
    assertTokenBytes(lexer, NinjaToken.identifier, "bad");
    assertError(lexer, "Bad $-escape (literal $ must be written as $$)", "$");

    NinjaLexer lexer2 = createLexer("$$$");
    assertTokenBytes(lexer2, NinjaToken.text, "$");
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
    assertTokenBytes(createLexer("build"), NinjaToken.build, null);
    assertTokenBytes(createLexer("rule"), NinjaToken.rule, null);
    assertTokenBytes(createLexer("default "), NinjaToken.default_, null);
    assertTokenBytes(createLexer("include"), NinjaToken.include, null);
    assertTokenBytes(createLexer("subninja\n"), NinjaToken.subninja, null);
    assertTokenBytes(createLexer("pool "), NinjaToken.pool, null);
  }

  @Test
  public void testIndent() {
    NinjaLexer lexer = createLexer(" a\nb\n  c   d   e\n ");
    // We want to know if there was a starting indent
    // (though we suppose to start with a line without indent)
    assertTokenBytes(lexer, NinjaToken.indent, null);
    assertTokenBytes(lexer, NinjaToken.identifier, "a");
    assertTokenBytes(lexer, NinjaToken.newline, null);
    assertTokenBytes(lexer, NinjaToken.identifier, "b");
    assertTokenBytes(lexer, NinjaToken.newline, null);
    assertTokenBytes(lexer, NinjaToken.indent, null);
    assertTokenBytes(lexer, NinjaToken.identifier, "c");
    assertTokenBytes(lexer, NinjaToken.identifier, "d");
    assertTokenBytes(lexer, NinjaToken.identifier, "e");
    assertTokenBytes(lexer, NinjaToken.newline, null);
    assertTokenBytes(lexer, NinjaToken.indent, null);
  }

  @Test
  public void testReadText() {
    NinjaLexer lexer = createLexer("my.var=Any text ^&%$@&!*: $:symbols$\n aa\nmy.var2");
    assertTokenBytes(lexer, NinjaToken.identifier, "my.var");
    assertTokenBytes(lexer, NinjaToken.equals, null);
    lexer.setExpectTextUntilEol(true);
    assertTokenBytes(lexer, NinjaToken.text, "Any");
    assertTokenBytes(lexer, NinjaToken.text, "text");
    assertTokenBytes(lexer, NinjaToken.text, "^&%");
    assertTokenBytes(lexer, NinjaToken.text, "$");
    assertTokenBytes(lexer, NinjaToken.text, "@&!*");
    assertTokenBytes(lexer, NinjaToken.colon, null);
    assertTokenBytes(lexer, NinjaToken.text, ":");
    assertTokenBytes(lexer, NinjaToken.text, "symbols");
    assertTokenBytes(lexer, NinjaToken.text, "aa");
    assertTokenBytes(lexer, NinjaToken.newline, null);
    assertTokenBytes(lexer, NinjaToken.identifier, "my.var2");
  }

  @Test
  public void testReadTextFragment() {
    NinjaLexer lexer = createLexer("my.var=Any text ^&%$@&!* $:symbols$\n aa\nmy.var2");
    assertTokenBytes(lexer, NinjaToken.identifier, "my.var");
    assertTokenBytes(lexer, NinjaToken.equals, null);

    assertThat(lexer.readTextFragment()).isEqualTo("Any text ^&%$@&!* $:symbols$\n aa"
        .getBytes(StandardCharsets.ISO_8859_1));
    assertTokenBytes(lexer, NinjaToken.newline, null);
    assertTokenBytes(lexer, NinjaToken.identifier, "my.var2");
  }

  @Test
  public void testUndo() {
    NinjaLexer lexer = createLexer("my.var=Any\n");
    assertTokenBytes(lexer, NinjaToken.identifier, "my.var");
    assertTokenBytes(lexer, NinjaToken.equals, null);
    assertTokenBytes(lexer, NinjaToken.identifier, "Any");
    assertTokenBytes(lexer, NinjaToken.newline, null);

    lexer.undo();
    assertTokenBytes(lexer, NinjaToken.newline, null);
    lexer.undo();
    lexer.undo();
    assertTokenBytes(lexer, NinjaToken.identifier, "Any");
    assertTokenBytes(lexer, NinjaToken.newline, null);
    lexer.undo();
    lexer.undo();
    lexer.undo();
    assertTokenBytes(lexer, NinjaToken.equals, null);
    assertTokenBytes(lexer, NinjaToken.identifier, "Any");
    assertTokenBytes(lexer, NinjaToken.newline, null);
    lexer.undo();
    lexer.undo();
    lexer.undo();
    lexer.undo();
    assertTokenBytes(lexer, NinjaToken.identifier, "my.var");
    lexer.undo();
    assertTokenBytes(lexer, NinjaToken.identifier, "my.var");
    assertTokenBytes(lexer, NinjaToken.equals, null);
    assertTokenBytes(lexer, NinjaToken.identifier, "Any");
    assertTokenBytes(lexer, NinjaToken.newline, null);
  }

  @Test
  public void testSpecialSymbols() {
    NinjaLexer lexer = createLexer("| || : = ");
    assertTokenBytes(lexer, NinjaToken.pipe, null);
    assertTokenBytes(lexer, NinjaToken.pipe2, null);
    assertTokenBytes(lexer, NinjaToken.colon, null);
    assertTokenBytes(lexer, NinjaToken.equals, null);
    assertTokenBytes(lexer, NinjaToken.eof, null);
    assertThat(lexer.hasNextToken()).isFalse();
  }

  @Test
  public void testZeroByte() {
    byte[] bytes = {'a', 0, 'b'};
    NinjaLexer lexer = new NinjaLexer(
        new ByteBufferFragment(ByteBuffer.wrap(bytes), 0, bytes.length));
    assertTokenBytes(lexer, NinjaToken.identifier, null);
    assertThat(lexer.hasNextToken()).isFalse();
  }

  private void assertTokenBytes(NinjaLexer lexer, NinjaToken token, @Nullable String text) {
    assertThat(lexer.hasNextToken()).isTrue();
    assertThat(lexer.nextToken()).isEqualTo(token);
    if (text != null) {
      assertThat(lexer.getTokenBytes()).isEqualTo(text.getBytes(StandardCharsets.ISO_8859_1));
    }
  }

  private NinjaLexer createLexer(String text) {
    ByteBuffer buffer = ByteBuffer.wrap(text.getBytes(StandardCharsets.ISO_8859_1));
    return new NinjaLexer(new ByteBufferFragment(buffer, 0, buffer.limit()));
  }
}
