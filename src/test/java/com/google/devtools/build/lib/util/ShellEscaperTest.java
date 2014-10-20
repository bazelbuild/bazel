// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.util;

import static com.google.devtools.build.lib.util.ShellEscaper.escapeString;
import static org.junit.Assert.assertEquals;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableSet;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.Arrays;
import java.util.Set;

/**
 * Tests for {@link ShellEscaper}.
 *
 * <p>Based on {@code com.google.io.base.shell.ShellUtilsTest}.
 */
@RunWith(JUnit4.class)
public class ShellEscaperTest {

  @Test
  public void shellEscape() throws Exception {
    assertEquals("''", escapeString(""));
    assertEquals("foo", escapeString("foo"));
    assertEquals("'foo bar'", escapeString("foo bar"));
    assertEquals("''\\''foo'\\'''", escapeString("'foo'"));
    assertEquals("'\\'\\''foo\\'\\'''", escapeString("\\'foo\\'"));
    assertEquals("'${filename%.c}.o'", escapeString("${filename%.c}.o"));
    assertEquals("'<html!>'", escapeString("<html!>"));
  }

  @Test
  public void escapeAll() throws Exception {
    Set<String> escaped = ImmutableSet.copyOf(
        ShellEscaper.escapeAll(Arrays.asList("foo", "@bar", "baz'qux")));
    assertEquals(ImmutableSet.of("foo", "@bar", "'baz'\\''qux'"), escaped);
  }

  @Test
  public void escapeJoinAllIntoAppendable() throws Exception {
    Appendable appendable = ShellEscaper.escapeJoinAll(
        new StringBuilder("initial"), Arrays.asList("foo", "$BAR"));
    assertEquals("initialfoo '$BAR'", appendable.toString());
  }

  @Test
  public void escapeJoinAllIntoAppendableWithCustomJoiner() throws Exception {
    Appendable appendable = ShellEscaper.escapeJoinAll(
        new StringBuilder("initial"), Arrays.asList("foo", "$BAR"), Joiner.on('|'));
    assertEquals("initialfoo|'$BAR'", appendable.toString());
  }

  @Test
  public void escapeJoinAll() throws Exception {
    String actual = ShellEscaper.escapeJoinAll(
        Arrays.asList("foo", "@echo:-", "100", "$US", "a b", "\"qu'ot'es\"", "\"quot\"", "\\"));
    assertEquals("foo @echo:- 100 '$US' 'a b' '\"qu'\\''ot'\\''es\"' '\"quot\"' '\\'", actual);
  }

  @Test
  public void escapeJoinAllWithCustomJoiner() throws Exception {
    String actual = ShellEscaper.escapeJoinAll(
        Arrays.asList("foo", "@echo:-", "100", "$US", "a b", "\"qu'ot'es\"", "\"quot\"", "\\"),
        Joiner.on('|'));
    assertEquals("foo|@echo:-|100|'$US'|'a b'|'\"qu'\\''ot'\\''es\"'|'\"quot\"'|'\\'", actual);
  }
}
