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

package com.google.devtools.build.lib.bazel.rules.ninja.parser;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Ascii;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Interner;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.bazel.rules.ninja.file.GenericParsingException;
import com.google.devtools.build.lib.bazel.rules.ninja.lexer.NinjaLexer;
import com.google.devtools.build.lib.bazel.rules.ninja.lexer.NinjaLexer.TextKind;
import com.google.devtools.build.lib.bazel.rules.ninja.lexer.NinjaToken;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaTarget.InputKind;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaTarget.InputOutputKind;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaTarget.OutputKind;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/** Ninja files parser. The types of tokens: {@link NinjaToken}. Ninja lexer: {@link NinjaLexer}. */
public class NinjaParserStep {
  /**
   * An interner for {@link PathFragment} instances for the inputs and outputs of {@link
   * NinjaTarget}.
   */
  // TODO(lberki): Make this non-static.
  // The reason why this field is static is that I haven't grokked yet what the lifetime of each
  // object in Ninja parsing is. Once I figure out which object has the lifetime of "exactly as long
  // as the parsing is running", I can just put this in a field of that object and plumb it here.
  private static final Interner<PathFragment> PATH_FRAGMENT_INTERNER =
      BlazeInterners.newWeakInterner();

  private final NinjaLexer lexer;

  public NinjaParserStep(NinjaLexer lexer) {
    this.lexer = lexer;
  }

  /** Parses variable at the current lexer position. */
  public Pair<String, NinjaVariableValue> parseVariable() throws GenericParsingException {
    String name = asString(parseExpected(NinjaToken.IDENTIFIER));
    parseExpected(NinjaToken.EQUALS);

    NinjaVariableValue value = parseVariableValue();
    return Pair.of(name, value);
  }

  @VisibleForTesting
  public NinjaVariableValue parseVariableValue() throws GenericParsingException {
    return Preconditions.checkNotNull(parseVariableValueImpl(true));
  }

  @Nullable
  private NinjaVariableValue parseVariableValueImpl(boolean noValueAsEmpty) {
    NinjaVariableValue.Builder varBuilder = NinjaVariableValue.builder();
    int previous = -1;
    while (lexer.hasNextToken()) {
      lexer.setExpectedTextKind(TextKind.TEXT);
      NinjaToken token = lexer.nextToken();
      if (NinjaToken.VARIABLE.equals(token)) {
        if (previous >= 0) {
          // add space interval between tokens
          varBuilder.addText(
              asString(lexer.getFragment().getBytes(previous, lexer.getLastStart())));
        }
        varBuilder.addVariable(normalizeVariableName(asString(lexer.getTokenBytes())));
      } else if (NinjaToken.TEXT.equals(token)
          || NinjaToken.ESCAPED_TEXT.equals(token)
          || NinjaToken.COLON.equals(token)) {
        // Add text together with the spaces between current and previous token.
        int start = previous >= 0 ? previous : lexer.getLastStart();
        String rawText = asString(lexer.getFragment().getBytes(start, lexer.getLastEnd()));
        String text = NinjaToken.ESCAPED_TEXT.equals(token) ? unescapeText(rawText) : rawText;
        varBuilder.addText(text);
      } else {
        lexer.undo();
        break;
      }
      previous = lexer.getLastEnd();
    }
    if (previous == -1) {
      // We read no value.
      if (noValueAsEmpty) {
        // Use empty string for value if specified by caller.
        return NinjaVariableValue.createPlainText("");
      }
      // Otherwise, return null to indicate there was no value.
      return null;
    }
    return varBuilder.build();
  }

  /**
   * Paths variable is a sequence of text and variable references until space, newline, eof or |
   * symbol.
   */
  @Nullable
  private NinjaVariableValue parsePathVariableValue() {
    NinjaVariableValue.Builder varBuilder = NinjaVariableValue.builder();
    int previous = -1;
    while (lexer.hasNextToken()) {
      lexer.setExpectedTextKind(TextKind.PATH);
      NinjaToken token = lexer.nextToken();
      if (previous >= 0 && lexer.getLastStart() != previous) {
        // no spaces.
        lexer.undo();
        break;
      }
      if (NinjaToken.VARIABLE.equals(token)) {
        varBuilder.addVariable(normalizeVariableName(asString(lexer.getTokenBytes())));
      } else if (NinjaToken.TEXT.equals(token) || NinjaToken.ESCAPED_TEXT.equals(token)) {
        String rawText = asString(lexer.getTokenBytes());
        String text = NinjaToken.ESCAPED_TEXT.equals(token) ? unescapeText(rawText) : rawText;
        varBuilder.addText(text);
      } else {
        lexer.undo();
        break;
      }
      previous = lexer.getLastEnd();
    }
    if (previous == -1) {
      // We read no value.
      return null;
    }
    return varBuilder.build();
  }

  private static String unescapeText(String text) {
    StringBuilder sb = new StringBuilder(text.length());
    for (int i = 0; i < text.length(); i++) {
      char ch = text.charAt(i);
      if (ch == '$') {
        Preconditions.checkState(i + 1 < text.length());
        sb.append(text.charAt(i + 1));
        i++;
      } else {
        sb.append(ch);
      }
    }
    return sb.toString();
  }

  public NinjaVariableValue parseIncludeStatement() throws GenericParsingException {
    return parseIncludeOrSubNinja(NinjaToken.INCLUDE);
  }

  public NinjaVariableValue parseSubNinjaStatement() throws GenericParsingException {
    return parseIncludeOrSubNinja(NinjaToken.SUBNINJA);
  }

  private NinjaVariableValue parseIncludeOrSubNinja(NinjaToken token)
      throws GenericParsingException {
    parseExpected(token);
    NinjaVariableValue value = parseVariableValueImpl(false);
    if (value == null) {
      throw new GenericParsingException(
          String.format("%s statement has no path.", Ascii.toLowerCase(token.name())));
    }
    if (lexer.hasNextToken()) {
      parseExpected(NinjaToken.NEWLINE);
      lexer.undo();
    }
    return value;
  }

  /** Parses Ninja rule at the current lexer position. */
  public NinjaRule parseNinjaRule() throws GenericParsingException {
    parseExpected(NinjaToken.RULE);
    String name = asString(parseExpected(NinjaToken.IDENTIFIER));

    ImmutableSortedMap.Builder<NinjaRuleVariable, NinjaVariableValue> variablesBuilder =
        ImmutableSortedMap.naturalOrder();
    variablesBuilder.put(NinjaRuleVariable.NAME, NinjaVariableValue.createPlainText(name));

    parseExpected(NinjaToken.NEWLINE);
    lexer.interpretPoolAsVariable();
    while (parseIndentOrFinishDeclaration()) {
      String variableName = asString(parseExpected(NinjaToken.IDENTIFIER));
      parseExpected(NinjaToken.EQUALS);
      NinjaVariableValue value = parseVariableValue();

      NinjaRuleVariable ninjaRuleVariable = NinjaRuleVariable.nullOrValue(variableName);
      if (ninjaRuleVariable == null) {
        throw new GenericParsingException(String.format("Unexpected variable '%s'", variableName));
      }
      variablesBuilder.put(ninjaRuleVariable, value);
      if (lexer.hasNextToken()) {
        parseExpected(NinjaToken.NEWLINE);
      }
    }
    return new NinjaRule(variablesBuilder.build());
  }

  /** Parses Ninja pool at the current lexer position. */
  public NinjaPool parseNinjaPool() throws GenericParsingException {
    // TODO: consider using generics to condense this and parseNinjaRule into the same
    // method body.
    parseExpected(NinjaToken.POOL);
    String name = asString(parseExpected(NinjaToken.IDENTIFIER));

    ImmutableSortedMap.Builder<NinjaPoolVariable, NinjaVariableValue> variablesBuilder =
        ImmutableSortedMap.naturalOrder();
    variablesBuilder.put(NinjaPoolVariable.NAME, NinjaVariableValue.createPlainText(name));

    parseExpected(NinjaToken.NEWLINE);
    while (parseIndentOrFinishDeclaration()) {
      String variableName = asString(parseExpected(NinjaToken.IDENTIFIER));
      parseExpected(NinjaToken.EQUALS);
      NinjaVariableValue value = parseVariableValue();

      NinjaPoolVariable ninjaPoolVariable = NinjaPoolVariable.nullOrValue(variableName);
      if (ninjaPoolVariable == null) {
        throw new GenericParsingException(String.format("Unexpected variable '%s'", variableName));
      }
      variablesBuilder.put(ninjaPoolVariable, value);
      if (lexer.hasNextToken()) {
        parseExpected(NinjaToken.NEWLINE);
      }
    }
    return new NinjaPool(variablesBuilder.build());
  }

  private enum NinjaTargetParsingPart {
    OUTPUTS(OutputKind.USUAL, true),
    IMPLICIT_OUTPUTS(OutputKind.IMPLICIT, true),
    INPUTS(InputKind.USUAL, false),
    IMPLICIT_INPUTS(InputKind.IMPLICIT, false),
    ORDER_ONLY_INPUTS(InputKind.ORDER_ONLY, false),
    RULE_NAME(null, false),
    VARIABLES(null, false);

    @Nullable private final InputOutputKind inputOutputKind;
    private final boolean transitionRequired;

    NinjaTargetParsingPart(@Nullable InputOutputKind inputOutputKind, boolean transitionRequired) {
      this.inputOutputKind = inputOutputKind;
      this.transitionRequired = transitionRequired;
    }

    @Nullable
    public InputOutputKind getInputOutputKind() {
      return inputOutputKind;
    }

    public boolean isTransitionRequired() {
      return transitionRequired;
    }
  }

  /**
   * Mapping for changing the {@link NinjaTargetParsingPart} according to the next separator symbol.
   */
  private static final ImmutableSortedMap<
          NinjaTargetParsingPart, ImmutableSortedMap<NinjaToken, NinjaTargetParsingPart>>
      TARGET_PARTS_TRANSITIONS_MAP =
          ImmutableSortedMap.of(
              NinjaTargetParsingPart.OUTPUTS,
                  ImmutableSortedMap.of(
                      NinjaToken.PIPE, NinjaTargetParsingPart.IMPLICIT_OUTPUTS,
                      NinjaToken.COLON, NinjaTargetParsingPart.RULE_NAME),
              NinjaTargetParsingPart.IMPLICIT_OUTPUTS,
                  ImmutableSortedMap.of(NinjaToken.COLON, NinjaTargetParsingPart.RULE_NAME),
              NinjaTargetParsingPart.INPUTS,
                  ImmutableSortedMap.of(
                      NinjaToken.PIPE, NinjaTargetParsingPart.IMPLICIT_INPUTS,
                      NinjaToken.PIPE2, NinjaTargetParsingPart.ORDER_ONLY_INPUTS,
                      NinjaToken.NEWLINE, NinjaTargetParsingPart.VARIABLES),
              NinjaTargetParsingPart.IMPLICIT_INPUTS,
                  ImmutableSortedMap.of(
                      NinjaToken.PIPE2, NinjaTargetParsingPart.ORDER_ONLY_INPUTS,
                      NinjaToken.NEWLINE, NinjaTargetParsingPart.VARIABLES),
              NinjaTargetParsingPart.ORDER_ONLY_INPUTS,
                  ImmutableSortedMap.of(NinjaToken.NEWLINE, NinjaTargetParsingPart.VARIABLES));

  /**
   * Parses Ninja target using {@link NinjaScope} of the file, where it is defined, to expand
   * variables.
   */
  public NinjaTarget parseNinjaTarget(NinjaScope fileScope, int offset)
      throws GenericParsingException {
    NinjaTarget.Builder builder = NinjaTarget.builder(fileScope, offset);
    parseExpected(NinjaToken.BUILD);

    Map<InputOutputKind, List<NinjaVariableValue>> pathValuesMap =
        parseTargetDependenciesPart(builder);

    NinjaScope targetScope = parseTargetVariables(offset, fileScope, builder);

    // Variables from the build statement can be used in the input and output paths, so
    // we are using targetScope to resolve paths values.
    for (Map.Entry<InputOutputKind, List<NinjaVariableValue>> entry : pathValuesMap.entrySet()) {
      List<PathFragment> paths =
          entry.getValue().stream()
              .map(
                  value ->
                      PATH_FRAGMENT_INTERNER.intern(
                          PathFragment.create(
                              targetScope.getExpandedValue(Integer.MAX_VALUE, value))))
              .collect(Collectors.toList());
      InputOutputKind inputOutputKind = entry.getKey();
      if (inputOutputKind instanceof InputKind) {
        builder.addInputs((InputKind) inputOutputKind, paths);
      } else {
        builder.addOutputs((OutputKind) inputOutputKind, paths);
      }
    }

    return builder.build();
  }

  /**
   * We resolve build statement variables values, using the file scope: build statement variable
   * values can not refer to each other. Then we are constructing the target's {@link NinjaScope}
   * with already expanded variables; it will be used for resolving target's input and output paths
   * (which can also refer to file-level variables, so we better reuse resolve logic that we already
   * have in NinjaScope).
   *
   * <p>As we expand variable values, we are adding them to {@link NinjaTarget.Builder}.
   *
   * <p>Ninja targets can not refer to the rule's variables values, because the rule variables are
   * only expanded when the rule is used, and the rule is used for already parsed target. However,
   * target's variables can override values of rule's variables.
   *
   * @return Ninja scope for expanding input and output paths of that statement
   */
  private NinjaScope parseTargetVariables(
      int offset, NinjaScope fileScope, NinjaTarget.Builder builder)
      throws GenericParsingException {
    Map<String, List<Pair<Integer, String>>> expandedVariables = Maps.newHashMap();
    lexer.interpretPoolAsVariable();
    while (parseIndentOrFinishDeclaration()) {
      Pair<String, NinjaVariableValue> pair = parseVariable();
      String name = Preconditions.checkNotNull(pair.getFirst());
      NinjaVariableValue value = Preconditions.checkNotNull(pair.getSecond());
      String expandedValue = fileScope.getExpandedValue(offset, value);
      expandedVariables
          .computeIfAbsent(name, k -> Lists.newArrayList())
          .add(Pair.of(0, expandedValue));
      builder.addVariable(name, expandedValue);

      if (lexer.hasNextToken()) {
        parseExpected(NinjaToken.NEWLINE);
      }
    }
    return fileScope.createScopeFromExpandedValues(ImmutableSortedMap.copyOf(expandedVariables));
  }

  /**
   * Parse build statement dependencies part: output1..k [| implicit_output1..k]: rule input1..k [|
   * implicit_input1..k] [|| order_only_input1..k]
   */
  private Map<InputOutputKind, List<NinjaVariableValue>> parseTargetDependenciesPart(
      NinjaTarget.Builder builder) throws GenericParsingException {
    Map<InputOutputKind, List<NinjaVariableValue>> pathValuesMap = Maps.newHashMap();
    boolean ruleNameParsed = false;
    NinjaTargetParsingPart parsingPart = NinjaTargetParsingPart.OUTPUTS;
    while (lexer.hasNextToken() && !NinjaTargetParsingPart.VARIABLES.equals(parsingPart)) {
      if (NinjaTargetParsingPart.RULE_NAME.equals(parsingPart)) {
        ruleNameParsed = true;
        builder.setRuleName(asString(parseExpected(NinjaToken.IDENTIFIER)));
        parsingPart = NinjaTargetParsingPart.INPUTS;
        continue;
      }
      List<NinjaVariableValue> paths = parsePaths();
      if (paths.isEmpty() && !NinjaTargetParsingPart.INPUTS.equals(parsingPart)) {
        throw new GenericParsingException("Expected paths sequence");
      }
      if (!paths.isEmpty()) {
        pathValuesMap.put(Preconditions.checkNotNull(parsingPart.getInputOutputKind()), paths);
      }
      if (!lexer.hasNextToken()) {
        if (parsingPart.isTransitionRequired()) {
          throw new GenericParsingException("Unexpected end of target");
        }
        break;
      }
      NinjaToken lexicalSeparator = lexer.nextToken();
      parsingPart =
          Preconditions.checkNotNull(TARGET_PARTS_TRANSITIONS_MAP.get(parsingPart))
              .get(lexicalSeparator);

      if (parsingPart == null) {
        throw new GenericParsingException("Unexpected token: " + lexicalSeparator);
      }
    }
    if (!ruleNameParsed) {
      throw new GenericParsingException("Expected rule name");
    }
    Preconditions.checkState(
        !lexer.hasNextToken() || NinjaTargetParsingPart.VARIABLES.equals(parsingPart));
    return pathValuesMap;
  }

  private List<NinjaVariableValue> parsePaths() {
    List<NinjaVariableValue> result = Lists.newArrayList();
    NinjaVariableValue value;
    while (lexer.hasNextToken() && (value = parsePathVariableValue()) != null) {
      result.add(value);
    }
    return result;
  }

  @VisibleForTesting
  public static String normalizeVariableName(String raw) {
    // We start from 1 because it is always at least $ marker symbol in the beginning
    int start = 1;
    for (; start < raw.length(); start++) {
      char ch = raw.charAt(start);
      if (' ' != ch && '$' != ch && '{' != ch) {
        break;
      }
    }
    int end = raw.length() - 1;
    for (; end > start; end--) {
      char ch = raw.charAt(end);
      if (' ' != ch && '}' != ch) {
        break;
      }
    }
    return raw.substring(start, end + 1);
  }

  private static String asString(byte[] value) {
    return new String(value, StandardCharsets.ISO_8859_1);
  }

  /**
   * It is expected that indent is preceding to the identifier in the scoped variable declaration.
   * It can be, however, that it is just an empty line with spaces - in that case, we want to
   * interpret it as the finish of the currently parsed lexeme.
   *
   * @return true if indent was parsed and there is something different than the newline after it.
   */
  private boolean parseIndentOrFinishDeclaration() throws GenericParsingException {
    if (!lexer.hasNextToken()) {
      return false;
    }
    NinjaToken token = lexer.nextToken();
    boolean isIndent = NinjaToken.INDENT.equals(token);
    if (!isIndent && !NinjaToken.NEWLINE.equals(token)) {
      throwNotExpectedTokenError(NinjaToken.INDENT, token);
    }
    // Check for indent followed by newline, or end of file.
    if (lexer.hasNextToken()) {
      NinjaToken afterIndent = lexer.nextToken();
      lexer.undo();
      if (NinjaToken.NEWLINE.equals(afterIndent)) {
        return false;
      }
    } else {
      return false;
    }
    return isIndent;
  }

  private byte[] parseExpected(NinjaToken expectedToken) throws GenericParsingException {
    checkLexerHasNextToken(expectedToken);
    NinjaToken token = lexer.nextToken();
    if (!expectedToken.equals(token)) {
      throwNotExpectedTokenError(expectedToken, token);
    }
    return lexer.getTokenBytes();
  }

  private void throwNotExpectedTokenError(NinjaToken expectedToken, NinjaToken token)
      throws GenericParsingException {
    String actual =
        NinjaToken.ERROR.equals(token)
            ? String.format("error: '%s'", lexer.getError())
            : asString(token.getBytes());
    throw new GenericParsingException(
        String.format(
            "Expected %s, but got %s in fragment:\n%s\n",
            asString(expectedToken.getBytes()),
            actual,
            lexer.getFragment().getFragmentAround(lexer.getLastStart())));
  }

  private void checkLexerHasNextToken(NinjaToken expectedToken) throws GenericParsingException {
    if (!lexer.hasNextToken()) {
      String message;
      if (lexer.haveReadAnyTokens()) {
        message =
            String.format(
                "Expected %s after '%s' in fragment:\n%s\n",
                asString(expectedToken.getBytes()),
                asString(lexer.getTokenBytes()),
                lexer.getFragment().getFragmentAround(lexer.getLastStart()));
      } else {
        message =
            String.format(
                "Expected %s, but found no text to parse after fragment:\n%s\n",
                asString(expectedToken.getBytes()),
                lexer.getFragment().getFragmentAround(lexer.getLastStart()));
      }
      throw new GenericParsingException(message);
    }
  }
}
