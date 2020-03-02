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

package com.google.devtools.build.lib.bazel.rules.ninja.lexer;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.bazel.rules.ninja.file.ByteBufferFragment;
import com.google.devtools.build.lib.util.Pair;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

/** Ninja files lexer. The types of tokens: {@link NinjaToken}. */
public class NinjaLexer {
  // They all are having different first letter, let's use it.
  private static final ImmutableMap<Byte, NinjaToken> KEYWORD_MAP =
      // There is no #of() method for 6 key-value pairs.
      Stream.of(
              NinjaToken.BUILD,
              NinjaToken.RULE,
              NinjaToken.DEFAULT,
              NinjaToken.SUBNINJA,
              NinjaToken.INCLUDE,
              NinjaToken.POOL)
          .collect(ImmutableMap.toImmutableMap(token -> token.getBytes()[0], nt -> nt));

  private final ByteBufferFragment fragment;
  private NinjaLexerStep step;
  private final List<Pair<Integer, Integer>> ranges;
  private final List<NinjaToken> tokens;
  /** Flag to give a hint how letters should be interpreted (as text, identifier, path). */
  private TextKind expectedTextKind = TextKind.IDENTIFIER;

  private boolean interpretPoolAsVariable = false;

  /** @param fragment fragment to do the lexing on */
  public NinjaLexer(ByteBufferFragment fragment) {
    this.fragment = fragment;
    step = new NinjaLexerStep(fragment, 0);
    ranges = Lists.newArrayList();
    tokens = Lists.newArrayList();
  }

  /**
   * Returns true if following nextToken() call may produce meaningful token. However, it may happen
   * that nextToken() will only produce {@link NinjaToken#EOF}, {@link NinjaToken#ZERO} or {@link
   * NinjaToken#ERROR}.
   *
   * <p>It is an optimization here to check for 'seen' flags: nextToken() may return some meaningful
   * token, and at the same time already discover the end of file or zero byte.
   */
  public boolean hasNextToken() {
    return step.canAdvance();
  }

  /**
   * Returns {@link NinjaToken} type of the token for the next non-space and non-comment token
   * at/after current <code>position</code> position.
   */
  public NinjaToken nextToken() {
    Preconditions.checkState(step.canAdvance());
    while (step.canAdvance()) {
      // First byte is checked right in constructor.
      if (step.isSeenZero()) {
        return push(NinjaToken.ZERO);
      }
      byte b = step.startByte();
      switch (b) {
        case ' ':
        case '\t':
          step.skipSpaces();
          if (step.getPosition() == 0
              || NinjaToken.NEWLINE.equals(Iterables.getLast(tokens, null))) {
            return push(NinjaToken.INDENT);
          }
          break;
        case '\r':
          expectedTextKind = TextKind.IDENTIFIER;
          step.processLineFeedNewLine();
          return push(NinjaToken.NEWLINE);
        case '\n':
          expectedTextKind = TextKind.IDENTIFIER;
          return push(NinjaToken.NEWLINE);
        case '#':
          step.skipComment();
          break;
        case '=':
          if (TextKind.TEXT.equals(expectedTextKind)) {
            step.readText();
            return push(NinjaToken.TEXT);
          }
          return push(NinjaToken.EQUALS);
        case ':':
          return push(NinjaToken.COLON);
        case '|':
          if (TextKind.TEXT.equals(expectedTextKind)) {
            step.readText();
            return push(NinjaToken.TEXT);
          }
          if (step.tryReadDoublePipe()) {
            return push(NinjaToken.PIPE2);
          }
          return push(NinjaToken.PIPE);
        case '$':
          if (step.trySkipEscapedNewline()) {
            break;
          }
          if (step.tryReadVariableInBrackets() || step.tryReadSimpleVariable()) {
            return push(NinjaToken.VARIABLE);
          }
          if (step.tryReadEscapedLiteral()) {
            return push(NinjaToken.ESCAPED_TEXT);
          }
          step.forceError("Bad $-escape (literal $ must be written as $$)");
          return push(NinjaToken.ERROR);
        default:
          switch (expectedTextKind) {
            case TEXT:
              step.readText();
              return push(NinjaToken.TEXT);
            case PATH:
              step.readPath();
              return push(NinjaToken.TEXT);
            case IDENTIFIER:
              step.tryReadIdentifier();
              if (step.getError() == null) {
                byte[] bytes = step.getBytes();
                NinjaToken keywordToken = KEYWORD_MAP.get(bytes[0]);
                if (keywordToken != null
                    && !(interpretPoolAsVariable && NinjaToken.POOL.equals(keywordToken))
                    && Arrays.equals(keywordToken.getBytes(), bytes)) {
                  return push(keywordToken);
                }
              }
              return push(NinjaToken.IDENTIFIER);
          }
          throw new IllegalStateException();
      }
      if (step.canAdvance()) {
        step.ensureEnd();
        // For all skipping cases: move to the next step.
        step = step.nextStep();
      }
    }
    return push(NinjaToken.EOF);
  }

  /** Return the bytes of the token, returned by previous nextToken() call. */
  public byte[] getTokenBytes() {
    if (ranges.isEmpty()) {
      throw new IllegalStateException();
    }
    return fragment.getBytes(getLastStart(), getLastEnd());
  }

  private NinjaToken push(NinjaToken token) {
    step.ensureEnd();
    ranges.add(Pair.of(step.getStart(), step.getEnd()));
    tokens.add(token);
    if (step.getError() != null) {
      // Do not move in case of error.
      return NinjaToken.ERROR;
    }
    if (step.canAdvance()) {
      step = step.nextStep();
    }
    return token;
  }

  public boolean haveReadAnyTokens() {
    return !ranges.isEmpty();
  }

  public int getLastStart() {
    if (ranges.isEmpty()) {
      throw new IllegalStateException();
    }
    return Preconditions.checkNotNull(Iterables.getLast(ranges).getFirst());
  }

  public int getLastEnd() {
    if (ranges.isEmpty()) {
      throw new IllegalStateException();
    }
    return Preconditions.checkNotNull(Iterables.getLast(ranges).getSecond());
  }

  /** Give a hint how letters should be interpreted (as text, identifier, path). */
  public void setExpectedTextKind(TextKind expectedTextKind) {
    this.expectedTextKind = expectedTextKind;
  }

  /** When the keyword 'pool' is met, interpret it as identifier, not as a keyword. */
  public void interpretPoolAsVariable() {
    this.interpretPoolAsVariable = true;
  }

  /** Undo the previously read token. */
  public void undo() {
    Preconditions.checkState(ranges.size() == tokens.size());
    ranges.remove(ranges.size() - 1);
    tokens.remove(tokens.size() - 1);
    step = new NinjaLexerStep(fragment, ranges.isEmpty() ? 0 : getLastEnd());
    expectedTextKind = TextKind.IDENTIFIER;
  }

  public String getError() {
    return step.getError();
  }

  public ByteBufferFragment getFragment() {
    return fragment;
  }

  /**
   * Enum with variants of text fragments parsing: as identifier (most restricted set of symbols),
   * path (all spaces should be $-escaped, and | symbol has a special meaning), or text.
   */
  public enum TextKind {
    IDENTIFIER,
    PATH,
    TEXT
  }
}
