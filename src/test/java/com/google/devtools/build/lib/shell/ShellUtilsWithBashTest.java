// Copyright 2015 The Bazel Authors. All rights reserved.
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
import static com.google.devtools.build.lib.shell.ShellUtils.tokenize;
import static org.junit.Assert.fail;

import com.google.common.collect.Lists;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for ShellUtils that call out to Bash. */
@RunWith(JUnit4.class)
public class ShellUtilsWithBashTest {

  private void assertTokenizeIsDualToPrettyPrint(String... args) throws Exception {
    List<String> in = Arrays.asList(args);
    String shellCommand = prettyPrintArgv(in);

    // Assert that pretty-print is correct, i.e. dual to the actual /bin/sh
    // tokenization.  This test assumes no newlines in the input:
    String[] execArgs = {
      "/bin/sh",
      "-c",
      "for i in " + shellCommand + "; do echo \"$i\"; done" // tokenize, one word per line
    };
    String stdout = null;
    try {
      stdout = new String(new Command(execArgs, System.getenv()).execute().getStdout());
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
    assertThat(words).isEqualTo(in);

    // Assert that tokenize is dual to pretty-print:
    List<String> out = new ArrayList<>();
    try {
      tokenize(out, shellCommand);
    } finally {
      if (out.isEmpty()) { // i.e. an exception
        System.err.println(in);
      }
    }
    assertThat(out).isEqualTo(in);
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
