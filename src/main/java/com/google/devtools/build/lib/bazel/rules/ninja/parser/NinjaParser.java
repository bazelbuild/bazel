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

package com.google.devtools.build.lib.bazel.rules.ninja.parser;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableSortedMap;
import com.google.devtools.build.lib.bazel.rules.ninja.file.GenericParsingException;
import com.google.devtools.build.lib.bazel.rules.ninja.lexer.NinjaLexer;
import com.google.devtools.build.lib.bazel.rules.ninja.lexer.NinjaToken;
import com.google.devtools.build.lib.util.Pair;
import java.nio.charset.StandardCharsets;

/**
 * Ninja files parser.
 * The types of tokens: {@link NinjaToken}.
 * Ninja lexer: {@link NinjaLexer}.
 */
public class NinjaParser {
  private final NinjaLexer lexer;

  public NinjaParser(NinjaLexer lexer) {
    this.lexer = lexer;
  }

  /**
   * Parses variable at the current lexer position.
   */
  public Pair<String, NinjaVariableValue> parseVariable() throws GenericParsingException {
    String name = asString(parseExpected(NinjaToken.IDENTIFIER));
    parseExpected(NinjaToken.EQUALS);

    NinjaVariableValue value = parseVariableValue();
    return Pair.of(name, value);
  }

  private NinjaVariableValue parseVariableValue() throws GenericParsingException {
    // We are skipping starting spaces.
    int valueStart = -1;
    ImmutableSortedMap.Builder<String, Pair<Integer, Integer>> builder =
        ImmutableSortedMap.naturalOrder();
    while (lexer.hasNextToken()) {
      lexer.expectTextUntilEol();
      NinjaToken token = lexer.nextToken();
      if (NinjaToken.VARIABLE.equals(token)) {
        if (valueStart == -1) {
          valueStart = lexer.getLastStart();
        }
        builder.put(normalizeVariableName(asString(lexer.getTokenBytes())),
            Pair.of(lexer.getLastStart(), lexer.getLastEnd()));
      } else if (NinjaToken.TEXT.equals(token)) {
        if (valueStart == -1) {
          valueStart = lexer.getLastStart();
        }
      } else {
        lexer.undo();
        break;
      }
    }
    if (valueStart == -1) {
      // We read no value.
      throw new GenericParsingException(String.format("Variable has no value: '%s'",
          lexer.getFragment().toString()));
    }
    String text = asString(lexer.getFragment().getBytes(valueStart, lexer.getLastEnd()));
    return new NinjaVariableValue(text, builder.build());
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

  private String asString(byte[] value) {
    return new String(value, StandardCharsets.ISO_8859_1);
  }

  private byte[] parseExpected(NinjaToken expectedToken) {
    if (!lexer.hasNextToken()) {
      throw new IllegalStateException();
    }
    NinjaToken token = lexer.nextToken();
    if (!expectedToken.equals(token)) {
      throw new IllegalStateException();
    }
    return lexer.getTokenBytes();
  }
}
