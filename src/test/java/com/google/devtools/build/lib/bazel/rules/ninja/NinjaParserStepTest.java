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
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.bazel.rules.ninja.file.ByteBufferFragment;
import com.google.devtools.build.lib.bazel.rules.ninja.file.GenericParsingException;
import com.google.devtools.build.lib.bazel.rules.ninja.lexer.NinjaLexer;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaFileParseResult;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaParserStep;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaRule;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaRuleVariable;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaScope;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaTarget;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaVariableValue;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.function.Function;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link NinjaParserStep}. */
@RunWith(JUnit4.class)
public class NinjaParserStepTest {
  @Test
  public void testSimpleVariable() throws Exception {
    doTestSimpleVariable("a=b", "a", "b");
    doTestSimpleVariable("a=b\nc", "a", "b");
    doTestSimpleVariable("a=b # comment", "a", "b");
    doTestSimpleVariable("a.b.c =    some long:    value", "a.b.c", "some long:    value");
    doTestSimpleVariable("a_11_24-rt.15= ^&%=#@", "a_11_24-rt.15", "^&%=#@");
  }

  @Test
  public void testVariableParsingException() {
    doTestVariableParsingException(" ", "Expected identifier, but got indent in fragment:\n \n");
    doTestVariableParsingException("a", "Expected = after 'a' in fragment:\na\n");
    doTestVariableParsingException(
        "^a=",
        "Expected identifier, but got error: 'Symbol '^' is not allowed in the identifier, "
            + "the text fragment with the symbol:\n^a=\n' in fragment:\n^a=\n");
  }

  private static void doTestVariableParsingException(String text, String message) {
    GenericParsingException exception =
        assertThrows(GenericParsingException.class, () -> createParser(text).parseVariable());
    assertThat(exception).hasMessageThat().isEqualTo(message);
  }

  @Test
  public void testNoValue() throws Exception {
    doTestNoValue("a=");
    doTestNoValue("a=\u000018");
    doTestNoValue("a  =    ");
    doTestNoValue("a  =\nm");
    doTestNoValue("a  =    # 123");
  }

  @Test
  public void testWithVariablesInValue() throws Exception {
    doTestWithVariablesInValue("a=$a $b", "a", "${a} ${b}", ImmutableSortedSet.of("a", "b"));
    doTestWithVariablesInValue("a=a_$b_c", "a", "a_${b_c}", ImmutableSortedSet.of("b_c"));
    doTestWithVariablesInValue("a=$b a c", "a", "${b} a c", ImmutableSortedSet.of("b"));
    doTestWithVariablesInValue("a=a_$b c", "a", "a_${b} c", ImmutableSortedSet.of("b"));
    doTestWithVariablesInValue("a=a_${b.d}c", "a", "a_${b.d}c", ImmutableSortedSet.of("b.d"));
    doTestWithVariablesInValue(
        "e=a$b*c${ d }*18", "e", "a${b}*c${d}*18", ImmutableSortedSet.of("b", "d"));
    doTestWithVariablesInValue("e=a$b*${ b }", "e", "a${b}*${b}", ImmutableSortedSet.of("b"));
  }

  @Test
  public void testNormalizeVariableName() {
    assertThat(NinjaParserStep.normalizeVariableName("$a")).isEqualTo("a");
    assertThat(NinjaParserStep.normalizeVariableName("$a-b-c")).isEqualTo("a-b-c");
    assertThat(NinjaParserStep.normalizeVariableName("${abc_de-7}")).isEqualTo("abc_de-7");
    assertThat(NinjaParserStep.normalizeVariableName("${ a1.5}")).isEqualTo("a1.5");
    assertThat(NinjaParserStep.normalizeVariableName("${a1.5  }")).isEqualTo("a1.5");
  }

  @Test
  public void testInclude() throws Exception {
    NinjaVariableValue value1 = createParser("include x/multi words/z").parseIncludeStatement();
    assertThat(value1.getRawText()).isEqualTo("x/multi words/z");

    NinjaVariableValue value2 = createParser("subninja ${x}.ninja").parseSubNinjaStatement();
    assertThat(value2.getRawText()).isEqualTo("${x}.ninja");
    MockValueExpander expander = new MockValueExpander("###");
    assertThat(value2.getExpandedValue(expander)).isEqualTo("###x.ninja");
    assertThat(expander.getRequestedVariables()).containsExactly("x");
  }

  @Test
  public void testIncludeErrors() {
    GenericParsingException exception1 =
        assertThrows(
            GenericParsingException.class,
            () -> createParser("include x $").parseIncludeStatement());
    assertThat(exception1)
        .hasMessageThat()
        .isEqualTo(
            "Expected newline, but got error: "
                + "'Bad $-escape (literal $ must be written as $$)' in fragment:\ninclude x $\n");

    GenericParsingException exception2 =
        assertThrows(
            GenericParsingException.class, () -> createParser("include").parseIncludeStatement());
    assertThat(exception2).hasMessageThat().isEqualTo("include statement has no path.");

    GenericParsingException exception3 =
        assertThrows(
            GenericParsingException.class,
            () -> createParser("subninja  \nm").parseSubNinjaStatement());
    assertThat(exception3).hasMessageThat().isEqualTo("subninja statement has no path.");
  }

  @Test
  public void testNinjaRule() throws Exception {
    // Additionally test the situation when we get more line separators in the end.
    NinjaParserStep parser =
        createParser(
            "rule testRule  \n"
                + " command = executable --flag $TARGET $out && $POST_BUILD\n"
                + " description = Test rule for $TARGET\n"
                + " rspfile = $TARGET.in\n"
                + " deps = ${abc} $\n"
                + " ${cde}\n\n\n");
    NinjaRule ninjaRule = parser.parseNinjaRule();
    ImmutableSortedMap<NinjaRuleVariable, NinjaVariableValue> variables = ninjaRule.getVariables();
    assertThat(variables.keySet())
        .containsExactly(
            NinjaRuleVariable.NAME,
            NinjaRuleVariable.COMMAND,
            NinjaRuleVariable.DESCRIPTION,
            NinjaRuleVariable.RSPFILE,
            NinjaRuleVariable.DEPS);
    assertThat(variables.get(NinjaRuleVariable.NAME).getRawText()).isEqualTo("testRule");
    assertThat(variables.get(NinjaRuleVariable.DEPS).getRawText()).isEqualTo("${abc} $\n ${cde}");
    MockValueExpander expander = new MockValueExpander("###");
    assertThat(variables.get(NinjaRuleVariable.DEPS).getExpandedValue(expander))
        .isEqualTo("###abc $\n ###cde");
    assertThat(expander.getRequestedVariables()).containsExactly("abc", "cde");
  }

  @Test
  public void testNinjaRuleWithHash() throws Exception {
    // Additionally test the situation when we get more line separators in the end.
    NinjaParserStep parser =
        createParser(
            "rule testRule  \n"
                + " command = executable --flag $TARGET $out && sed -e 's/#.*$$//' -e '/^$$/d'\n"
                + " description = Test rule for $TARGET");
    NinjaRule ninjaRule = parser.parseNinjaRule();
    assertThat(ninjaRule.getVariables().get(NinjaRuleVariable.COMMAND).getRawText())
        // Variables are wrapped with {} by print function, $$ escape sequence is unescaped.
        .isEqualTo("executable --flag ${TARGET} ${out} && sed -e 's/#.*$//' -e '/^$/d'");
  }

  @Test
  public void testVariableWithoutValue() throws Exception {
    NinjaParserStep parser =
        createParser(
            "rule testRule  \n"
                + " command = executable --flag $TARGET $out && $POST_BUILD\n"
                + " description =\n");
    NinjaRule ninjaRule = parser.parseNinjaRule();
    ImmutableSortedMap<NinjaRuleVariable, NinjaVariableValue> variables = ninjaRule.getVariables();
    assertThat(variables.keySet())
        .containsExactly(
            NinjaRuleVariable.NAME, NinjaRuleVariable.COMMAND, NinjaRuleVariable.DESCRIPTION);
    assertThat(variables.get(NinjaRuleVariable.NAME).getRawText()).isEqualTo("testRule");
    assertThat(variables.get(NinjaRuleVariable.DESCRIPTION).getRawText()).isEmpty();
  }

  @Test
  public void testNinjaRuleParsingException() {
    doTestNinjaRuleParsingException(
        "rule testRule extra-word\n",
        String.join(
            "\n",
            "Expected newline, but got identifier in fragment:",
            "rule testRule extra-word",
            "",
            ""));
    doTestNinjaRuleParsingException(
        "rule testRule\ncommand =",
        String.join(
            "\n",
            "Expected indent, but got identifier in fragment:",
            "rule testRule",
            "command =",
            ""));
    doTestNinjaRuleParsingException(
        "rule testRule\n ^custom = a",
        String.join(
            "\n",
            "Expected identifier, but got error: 'Symbol '^' is not allowed in the identifier, "
                + "the text fragment with the symbol:",
            "rule testRule",
            " ^custom = a",
            "' in fragment:",
            "rule testRule",
            " ^custom = a",
            ""));
    doTestNinjaRuleParsingException("rule testRule\n custom = a", "Unexpected variable 'custom'");
  }

  @Test
  public void testNinjaTargets() throws Exception {
    // Additionally test the situation when the target does not have the variables section and
    // we get more line separators in the end.
    NinjaTarget target = parseNinjaTarget("build output: command input\n\n");
    assertThat(target.getRuleName()).isEqualTo("command");
    assertThat(target.getOutputs()).containsExactly(PathFragment.create("output"));
    assertThat(target.getUsualInputs()).containsExactly(PathFragment.create("input"));

    NinjaTarget target1 =
        parseNinjaTarget("build o1 o2 | io1 io2: command i1 i2 | ii1 ii2 || ooi1 ooi2");
    assertThat(target1.getRuleName()).isEqualTo("command");
    assertThat(target1.getOutputs())
        .containsExactly(PathFragment.create("o1"), PathFragment.create("o2"));
    assertThat(target1.getImplicitOutputs())
        .containsExactly(PathFragment.create("io1"), PathFragment.create("io2"));
    assertThat(target1.getUsualInputs())
        .containsExactly(PathFragment.create("i1"), PathFragment.create("i2"));
    assertThat(target1.getImplicitInputs())
        .containsExactly(PathFragment.create("ii1"), PathFragment.create("ii2"));
    assertThat(target1.getOrderOnlyInputs())
        .containsExactly(PathFragment.create("ooi1"), PathFragment.create("ooi2"));

    NinjaTarget target2 = parseNinjaTarget("build output: phony");
    assertThat(target2.getRuleName()).isEqualTo("phony");
    assertThat(target2.getOutputs()).containsExactly(PathFragment.create("output"));

    NinjaTarget target3 = parseNinjaTarget("build output: command $\n || order-only-input");
    assertThat(target3.getRuleName()).isEqualTo("command");
    assertThat(target3.getOutputs()).containsExactly(PathFragment.create("output"));
    assertThat(target3.getOrderOnlyInputs())
        .containsExactly(PathFragment.create("order-only-input"));
  }

  @Test
  public void testNinjaTargetParsingErrors() {
    testNinjaTargetParsingError("build xxx", "Unexpected end of target");
    testNinjaTargetParsingError("build xxx yyy:", "Expected rule name");
    testNinjaTargetParsingError("build xxx || yyy: command", "Unexpected token: PIPE2");
    testNinjaTargetParsingError("build xxx: command :", "Unexpected token: COLON");
    testNinjaTargetParsingError("build xxx: command | || a", "Expected paths sequence");
  }

  @Test
  public void testNinjaTargetsWithVariables() throws Exception {
    NinjaFileParseResult parseResult = new NinjaFileParseResult();
    parseResult.addVariable("output", 1, NinjaVariableValue.createPlainText("out123"));
    parseResult.addVariable("input", 2, NinjaVariableValue.createPlainText("in123"));

    NinjaScope scope = new NinjaScope();
    parseResult.expandIntoScope(scope, Maps.newHashMap());

    // Variables, defined inside build statement, are used for input and output paths,
    // but not for the values of the other variables.
    // Test it.
    NinjaTarget target =
        createParser(
                "build $output : command $input $dir/abcde\n"
                    + "  dir = def$input\n  empty = '$dir'")
            .parseNinjaTarget(scope, 5);
    assertThat(target.getRuleName()).isEqualTo("command");
    assertThat(target.getOutputs()).containsExactly(PathFragment.create("out123"));
    assertThat(target.getUsualInputs())
        .containsExactly(PathFragment.create("in123"), PathFragment.create("defin123/abcde"));
    assertThat(target.getVariables())
        .containsExactlyEntriesIn(ImmutableSortedMap.of("dir", "defin123", "empty", "''"));
  }

  @Test
  public void testPseudoCyclesOfVariables() throws Exception {
    NinjaFileParseResult parseResult = new NinjaFileParseResult();
    parseResult.addVariable(
        "output", 1, NinjaVariableValue.builder().addText("'out'").addVariable("input").build());
    parseResult.addVariable(
        "input", 2, NinjaVariableValue.builder().addText("'in'").addVariable("output").build());
    NinjaScope scope = new NinjaScope();
    parseResult.expandIntoScope(scope, Maps.newHashMap());
    assertThat(scope.findExpandedVariable(3, "input")).isEqualTo("'in''out'");
    assertThat(scope.findExpandedVariable(3, "output")).isEqualTo("'out'");
  }

  @Test
  public void testNinjaTargetsPathWithEscapedSpace() throws Exception {
    NinjaTarget target = parseNinjaTarget("build output : command input$ with$ space other");
    assertThat(target.getRuleName()).isEqualTo("command");
    assertThat(target.getOutputs()).containsExactly(PathFragment.create("output"));
    assertThat(target.getUsualInputs())
        .containsExactly(PathFragment.create("input with space"), PathFragment.create("other"));
  }

  @Test
  public void testNinjaTargetWithScope() throws Exception {
    NinjaTarget target = parseNinjaTarget("build output : command input\n  pool = abc\n");
    assertThat(target.getRuleName()).isEqualTo("command");
    assertThat(target.getOutputs()).containsExactly(PathFragment.create("output"));
    assertThat(target.getUsualInputs()).containsExactly(PathFragment.create("input"));
  }

  private static void testNinjaTargetParsingError(String text, String error) {
    GenericParsingException exception =
        assertThrows(GenericParsingException.class, () -> parseNinjaTarget(text));
    assertThat(exception).hasMessageThat().isEqualTo(error);
  }

  private static NinjaTarget parseNinjaTarget(String text) throws Exception {
    NinjaScope fileScope = new NinjaScope();
    return createParser(text).parseNinjaTarget(fileScope, 0);
  }

  private static void doTestNinjaRuleParsingException(String text, String message) {
    GenericParsingException exception =
        assertThrows(GenericParsingException.class, () -> createParser(text).parseNinjaRule());
    assertThat(exception).hasMessageThat().isEqualTo(message);
  }

  private static void doTestSimpleVariable(String text, String name, String value)
      throws Exception {
    NinjaParserStep parser = createParser(text);
    Pair<String, NinjaVariableValue> variable = parser.parseVariable();
    assertThat(variable.getFirst()).isEqualTo(name);
    assertThat(variable.getSecond()).isNotNull();
    assertThat(variable.getSecond().getRawText()).isEqualTo(value);

    MockValueExpander expander = new MockValueExpander("###");
    assertThat(variable.getSecond().getExpandedValue(expander)).isEqualTo(value);
    assertThat(expander.getRequestedVariables()).isEmpty();
  }

  private static void doTestNoValue(String text) throws Exception {
    NinjaParserStep parser = createParser(text);
    NinjaVariableValue value = parser.parseVariable().getSecond();
    assertThat(value).isNotNull();
    assertThat(value.getRawText()).isEmpty();
  }

  private static void doTestWithVariablesInValue(
      String text, String name, String value, ImmutableSortedSet<String> expectedVars)
      throws Exception {
    NinjaParserStep parser = createParser(text);
    Pair<String, NinjaVariableValue> variable = parser.parseVariable();
    assertThat(variable.getFirst()).isEqualTo(name);
    assertThat(variable.getSecond()).isNotNull();
    assertThat(variable.getSecond().getRawText()).isEqualTo(value);

    MockValueExpander expander = new MockValueExpander("###");
    assertThat(variable.getSecond().getExpandedValue(expander)).contains("###");
    assertThat(expander.getRequestedVariables()).containsExactlyElementsIn(expectedVars);
  }

  private static NinjaParserStep createParser(String text) {
    ByteBuffer buffer = ByteBuffer.wrap(text.getBytes(StandardCharsets.ISO_8859_1));
    NinjaLexer lexer = new NinjaLexer(new ByteBufferFragment(buffer, 0, buffer.limit()));
    return new NinjaParserStep(lexer);
  }

  private static class MockValueExpander implements Function<String, String> {
    private final ImmutableSortedSet.Builder<String> setBuilder;
    private final String prefix;

    private MockValueExpander(String prefix) {
      this.prefix = prefix;
      setBuilder = ImmutableSortedSet.naturalOrder();
    }

    @Override
    public String apply(String s) {
      setBuilder.add(s);
      return prefix + s;
    }

    public ImmutableSortedSet<String> getRequestedVariables() {
      return setBuilder.build();
    }
  }
}
