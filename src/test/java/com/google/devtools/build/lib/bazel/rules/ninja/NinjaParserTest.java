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
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Range;
import com.google.devtools.build.lib.bazel.rules.ninja.file.ByteBufferFragment;
import com.google.devtools.build.lib.bazel.rules.ninja.file.GenericParsingException;
import com.google.devtools.build.lib.bazel.rules.ninja.lexer.NinjaLexer;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaParser;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaRule;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaRuleVariable;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaVariableValue;
import com.google.devtools.build.lib.collect.ImmutableSortedKeyListMultimap;
import com.google.devtools.build.lib.util.Pair;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaParser}. */
@RunWith(JUnit4.class)
public class NinjaParserTest {
  @Test
  public void testSimpleVariable() throws Exception {
    doTestSimpleVariable("a=b", "a", "b");
    doTestSimpleVariable("a=b\nc", "a", "b");
    doTestSimpleVariable("a=b # comment", "a", "b");
    doTestSimpleVariable("a.b.c =    some long    value", "a.b.c", "some long    value");
    doTestSimpleVariable("a_11_24-rt.15= ^&%$#@", "a_11_24-rt.15", "^&%$");
  }

  @Test
  public void testVariableParsingException() {
    doTestVariableParsingException(" ", "Expected identifier, but got indent");
    doTestVariableParsingException("a", "Expected = after 'a'");
    doTestVariableParsingException("a=:", "Variable 'a' has no value.");
    doTestVariableParsingException(
        "^a=", "Expected identifier, but got error: 'Symbol is not allowed in the identifier.'");
  }

  private static void doTestVariableParsingException(String text, String message) {
    GenericParsingException exception =
        assertThrows(GenericParsingException.class, () -> createParser(text).parseVariable());
    assertThat(exception).hasMessageThat().isEqualTo(message);
  }

  @Test
  public void testNoValue() {
    doTestNoValue("a=");
    doTestNoValue("a=\u000018");
    doTestNoValue("a  =    ");
    doTestNoValue("a  =\nm");
    doTestNoValue("a  =    # 123");
  }

  @Test
  public void testWithVariablesInValue() throws Exception {
    doTestWithVariablesInValue("a=$a $b", "a", "$a $b", expectedVariables("a", 2, 4, "b", 5, 7));
    doTestWithVariablesInValue("a=a_$b_c", "a", "a_$b_c", expectedVariables("b_c", 4, 8));
    doTestWithVariablesInValue("a=$b a c", "a", "$b a c", expectedVariables("b", 2, 4));
    doTestWithVariablesInValue("a=a_$b c", "a", "a_$b c", expectedVariables("b", 4, 6));
    doTestWithVariablesInValue("a=a_${b.d}c", "a", "a_${b.d}c", expectedVariables("b.d", 4, 10));
    doTestWithVariablesInValue(
        "e=a$b*c${ d }*18", "e", "a$b*c${ d }*18", expectedVariables("b", 3, 5, "d", 7, 13));
    doTestWithVariablesInValue(
        "e=a$b*${ b }", "e", "a$b*${ b }", expectedVariables("b", 3, 5, "b", 6, 12));
  }

  @Test
  public void testNormalizeVariableName() {
    assertThat(NinjaParser.normalizeVariableName("$a")).isEqualTo("a");
    assertThat(NinjaParser.normalizeVariableName("$a-b-c")).isEqualTo("a-b-c");
    assertThat(NinjaParser.normalizeVariableName("${abc_de-7}")).isEqualTo("abc_de-7");
    assertThat(NinjaParser.normalizeVariableName("${ a1.5}")).isEqualTo("a1.5");
    assertThat(NinjaParser.normalizeVariableName("${a1.5  }")).isEqualTo("a1.5");
  }

  @Test
  public void testNinjaRule() throws Exception {
    NinjaParser parser =
        createParser(
            "rule testRule  \n"
                + " command = executable --flag $TARGET $out && $POST_BUILD\n"
                + " description = Test rule for $TARGET\n"
                + " rspfile = $TARGET.in\n"
                + " deps = ${abc} $\n"
                + " ${cde}\n");
    NinjaRule ninjaRule = parser.parseNinjaRule();
    ImmutableSortedMap<NinjaRuleVariable, NinjaVariableValue> variables = ninjaRule.getVariables();
    assertThat(variables.keySet())
        .containsExactly(
            NinjaRuleVariable.NAME,
            NinjaRuleVariable.COMMAND,
            NinjaRuleVariable.DESCRIPTION,
            NinjaRuleVariable.RSPFILE,
            NinjaRuleVariable.DEPS);
    assertThat(variables.get(NinjaRuleVariable.NAME).getText()).isEqualTo("testRule");
    assertThat(variables.get(NinjaRuleVariable.DEPS).getText()).isEqualTo("${abc} $\n ${cde}");
  }

  @Test
  public void testNinjaRuleParsingException() {
    doTestNinjaRuleParsingException(
        "rule testRule extra-word\n", "Expected newline, but got identifier");
    doTestNinjaRuleParsingException(
        "rule testRule\n command =", "Variable 'command' has no value.");
    doTestNinjaRuleParsingException(
        "rule testRule\ncommand =", "Expected indent, but got identifier");
    doTestNinjaRuleParsingException(
        "rule testRule\n ^custom = a",
        "Expected identifier, but got error: 'Symbol is not allowed in the identifier.'");
    doTestNinjaRuleParsingException("rule testRule\n custom = a", "Unexpected variable 'custom'");
  }

  private static void doTestNinjaRuleParsingException(String text, String message) {
    GenericParsingException exception =
        assertThrows(GenericParsingException.class, () -> createParser(text).parseNinjaRule());
    assertThat(exception).hasMessageThat().isEqualTo(message);
  }

  private static void doTestSimpleVariable(String text, String name, String value)
      throws GenericParsingException {
    NinjaParser parser = createParser(text);
    Pair<String, NinjaVariableValue> variable = parser.parseVariable();
    assertThat(variable.getFirst()).isEqualTo(name);
    assertThat(variable.getSecond()).isNotNull();
    assertThat(variable.getSecond().getText()).isEqualTo(value);
    assertThat(variable.getSecond().getVariables()).isEmpty();
  }

  private static void doTestNoValue(String text) {
    NinjaParser parser = createParser(text);
    GenericParsingException exception =
        assertThrows(GenericParsingException.class, parser::parseVariable);
    assertThat(exception).hasMessageThat().isEqualTo("Variable 'a' has no value.");
  }

  private static ImmutableSortedKeyListMultimap<String, Range<Integer>> expectedVariables(
      String name, int start, int end) {
    return ImmutableSortedKeyListMultimap.<String, Range<Integer>>builder()
        .put(name, Range.openClosed(start, end))
        .build();
  }

  private static ImmutableSortedKeyListMultimap<String, Range<Integer>> expectedVariables(
      String name1, int start1, int end1, String name2, int start2, int end2) {
    return ImmutableSortedKeyListMultimap.<String, Range<Integer>>builder()
        .put(name1, Range.openClosed(start1, end1))
        .put(name2, Range.openClosed(start2, end2))
        .build();
  }

  private static void doTestWithVariablesInValue(
      String text,
      String name,
      String value,
      ImmutableSortedKeyListMultimap<String, Range<Integer>> expectedVars)
      throws GenericParsingException {
    NinjaParser parser = createParser(text);
    Pair<String, NinjaVariableValue> variable = parser.parseVariable();
    assertThat(variable.getFirst()).isEqualTo(name);
    assertThat(variable.getSecond()).isNotNull();
    assertThat(variable.getSecond().getText()).isEqualTo(value);

    ImmutableSortedKeyListMultimap<String, Range<Integer>> variables =
        variable.getSecond().getVariables();
    assertThat(variables).containsExactlyEntriesIn(expectedVars);
  }

  private static NinjaParser createParser(String text) {
    ByteBuffer buffer = ByteBuffer.wrap(text.getBytes(StandardCharsets.ISO_8859_1));
    NinjaLexer lexer = new NinjaLexer(new ByteBufferFragment(buffer, 0, buffer.limit()));
    return new NinjaParser(lexer);
  }
}
