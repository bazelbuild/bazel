// Copyright 2015 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.shell;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.shell.ShellUtils.prettyPrintArgv;
import static com.google.devtools.build.lib.shell.ShellUtils.shellEscape;
import static com.google.devtools.build.lib.shell.ShellUtils.tokenize;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import com.google.common.collect.Lists;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Tests for ShellUtils.
 */
@RunWith(JUnit4.class)
public class ShellUtilsTest {

  @Test
  public void testShellEscape() throws Exception {
    assertEquals("''", shellEscape(""));
    assertEquals("foo", shellEscape("foo"));
    assertEquals("'foo bar'", shellEscape("foo bar"));
    assertEquals("''\\''foo'\\'''", shellEscape("'foo'"));
    assertEquals("'\\'\\''foo\\'\\'''", shellEscape("\\'foo\\'"));
    assertEquals("'${filename%.c}.o'", shellEscape("${filename%.c}.o"));
    assertEquals("'<html!>'", shellEscape("<html!>"));
  }

  @Test
  public void testPrettyPrintArgv() throws Exception {
    assertEquals("echo '$US' 100",
                 prettyPrintArgv(Arrays.asList("echo", "$US", "100")));
  }

  private void assertTokenize(String copts, String... expectedTokens)
      throws Exception {
    List<String> actualTokens = new ArrayList<>();
    tokenize(actualTokens, copts);
    assertEquals(Arrays.asList(expectedTokens), actualTokens);
  }

  @Test
  public void testTokenize() throws Exception {
    assertTokenize("-DASMV", "-DASMV");
    assertTokenize("-DNO_UNDERLINE", "-DNO_UNDERLINE");
    assertTokenize("-DASMV -DNO_UNDERLINE",
                   "-DASMV", "-DNO_UNDERLINE");
    assertTokenize("-DDES_LONG=\"unsigned int\" -wd310",
                   "-DDES_LONG=unsigned int", "-wd310");
    assertTokenize("-Wno-write-strings -Wno-pointer-sign "
                   + "-Wno-unused-variable -Wno-pointer-to-int-cast",
                   "-Wno-write-strings",
                   "-Wno-pointer-sign",
                   "-Wno-unused-variable",
                   "-Wno-pointer-to-int-cast");
  }

  @Test
  public void testTokenizeOnNestedQuotation() throws Exception {
    assertTokenize("-Dfoo='foo\"bar' -Dwiz",
                   "-Dfoo=foo\"bar",
                   "-Dwiz");
    assertTokenize("-Dfoo=\"foo'bar\" -Dwiz",
                   "-Dfoo=foo'bar",
                   "-Dwiz");
  }

  @Test
  public void testTokenizeOnBackslashEscapes() throws Exception {
    // This would be easier to grok if we forked+exec'd a shell.

    assertTokenize("-Dfoo=\\'foo -Dbar", // \' not quoted -> '
                   "-Dfoo='foo",
                   "-Dbar");
    assertTokenize("-Dfoo=\\\"foo -Dbar", // \" not quoted -> "
                   "-Dfoo=\"foo",
                   "-Dbar");
    assertTokenize("-Dfoo=\\\\foo -Dbar", // \\ not quoted -> \
                   "-Dfoo=\\foo",
                   "-Dbar");

    assertTokenize("-Dfoo='\\'foo -Dbar", // \' single quoted -> \, close quote
                   "-Dfoo=\\foo",
                   "-Dbar");
    assertTokenize("-Dfoo='\\\"foo' -Dbar", // \" single quoted -> \"
                   "-Dfoo=\\\"foo",
                   "-Dbar");
    assertTokenize("-Dfoo='\\\\foo' -Dbar", // \\ single quoted -> \\
                   "-Dfoo=\\\\foo",
                   "-Dbar");

    assertTokenize("-Dfoo=\"\\'foo\" -Dbar", // \' double quoted -> \'
                   "-Dfoo=\\'foo",
                   "-Dbar");
    assertTokenize("-Dfoo=\"\\\"foo\" -Dbar", // \" double quoted -> "
                   "-Dfoo=\"foo",
                   "-Dbar");
    assertTokenize("-Dfoo=\"\\\\foo\" -Dbar", // \\ double quoted -> \
                   "-Dfoo=\\foo",
                   "-Dbar");
  }

  private void assertTokenizeFails(String copts, String expectedError) {
    try {
      tokenize(new ArrayList<String>(), copts);
      fail();
    } catch (ShellUtils.TokenizationException e) {
      assertThat(e).hasMessage(expectedError);
    }
  }

  @Test
  public void testTokenizeEmptyString() throws Exception {
    assertTokenize("");
  }

  @Test
  public void testTokenizeFailsOnUnterminatedQuotation() {
    assertTokenizeFails("-Dfoo=\"bar", "unterminated quotation");
    assertTokenizeFails("-Dfoo='bar", "unterminated quotation");
    assertTokenizeFails("-Dfoo=\"b'ar", "unterminated quotation");
  }

  private void assertTokenizeIsDualToPrettyPrint(String... args) throws Exception {
    List<String> in = Arrays.asList(args);
    String shellCommand = prettyPrintArgv(in);

    // Assert that pretty-print is correct, i.e. dual to the actual /bin/sh
    // tokenization.  This test assumes no newlines in the input:
    String[] execArgs =  {
      "/bin/sh",
      "-c",
      "for i in " + shellCommand + "; do echo \"$i\"; done"  // tokenize, one word per line
    };
    String stdout = null;
    try {
      stdout = new String(new Command(execArgs).execute().getStdout());
    } catch (Exception e) {
      fail("/bin/sh failed:\n" + in + "\n" + shellCommand + "\n" + e.getMessage());
    }
    // We can't use stdout.split("\n") here,
    // because String.split() ignores trailing empty strings.
    ArrayList<String> words = Lists.newArrayList();
    int index;
    while ((index = stdout.indexOf('\n')) >= 0) {
      words.add(stdout.substring(0, index));
      stdout = stdout.substring(index + 1);
    }
    assertEquals(in, words);

    // Assert that tokenize is dual to pretty-print:
    List<String> out = new ArrayList<>();
    try {
      tokenize(out, shellCommand);
    } finally {
      if (out.isEmpty()) { // i.e. an exception
        System.err.println(in);
      }
    }
    assertEquals(in, out);
  }

  @Test
  public void testTokenizeIsDualToPrettyPrint() throws Exception {
    // tokenize() is the inverse of prettyPrintArgv().  (However, the reverse
    // is not true, since there are many ways to escape the same string,
    // e.g. "foo" and 'foo'.)

    assertTokenizeIsDualToPrettyPrint("foo");
    assertTokenizeIsDualToPrettyPrint("foo bar");
    assertTokenizeIsDualToPrettyPrint("foo bar", "wiz");
    assertTokenizeIsDualToPrettyPrint("'foo'");
    assertTokenizeIsDualToPrettyPrint("\\'foo\\'");
    assertTokenizeIsDualToPrettyPrint("${filename%.c}.o");
    assertTokenizeIsDualToPrettyPrint("<html!>");

    assertTokenizeIsDualToPrettyPrint("");
    assertTokenizeIsDualToPrettyPrint("!@#$%^&*()");
    assertTokenizeIsDualToPrettyPrint("x'y\" z");
  }
}
