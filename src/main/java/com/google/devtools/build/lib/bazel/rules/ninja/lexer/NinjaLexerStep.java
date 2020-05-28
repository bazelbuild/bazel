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
import com.google.common.collect.ImmutableSortedSet;
import com.google.devtools.build.lib.bazel.rules.ninja.file.FileFragment;
import java.nio.charset.StandardCharsets;
import java.util.function.Predicate;

/**
 * Helper class for {@link NinjaLexer}. Contains methods for reading Ninja tokens.
 *
 * <p>Start position for reading is fixed. Mutable state includes the end position, offset in case
 * of escaped symbol, error text if a lexing error occurred, and the flag indicating if zero byte
 * was read. (Zero byte determines the end of the file.)
 *
 * <p>Intended to be used like: <code>
 * NinjaLexerStep step = new NinjaLexerStep(fragment, 0);
 * while (step.hasNext()) {
 *   byte b = step.startByte();
 *   // if/switch, then:
 *   step.skipXXX();
 *   // or
 *   step.tryXXX();
 *   // read the end position and error text
 *   if (step.hasNext()) {
 *     step = nextStep();
 *   }
 * }
 * </code>
 */
public class NinjaLexerStep {
  private static final ImmutableSortedSet<Byte> IDENTIFIER_SYMBOLS =
      createByteSet("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-");
  private static final byte[] TEXT_STOPPERS = createByteArray("\n\r \t$:\u0000");
  // We allow # symbol in the path, so the comment on the line with path can only start with space.
  private static final byte[] PATH_STOPPERS = createByteArray("\n\r \t$:|\u0000");

  private static byte[] createByteArray(String variants) {
    byte[] bytes = variants.getBytes(StandardCharsets.ISO_8859_1);
    return bytes;
  }

  private static ImmutableSortedSet<Byte> createByteSet(String variants) {
    ImmutableSortedSet.Builder<Byte> builder = ImmutableSortedSet.naturalOrder();
    byte[] bytes = variants.getBytes(StandardCharsets.ISO_8859_1);
    for (byte b : bytes) {
      builder.add(b);
    }
    return builder.build();
  }

  private final FileFragment fragment;
  private final int position;

  private boolean seenZero;
  private String error;
  private int end;

  /**
   * @param position start of the step inside a fragment; must point to a symbol inside fragment.
   */
  public NinjaLexerStep(FileFragment fragment, int position) {
    Preconditions.checkState(position < fragment.length());
    this.fragment = fragment;
    this.position = position;
    end = -1;
    seenZero = position < fragment.length() && (0 == fragment.byteAt(position));
  }

  public byte startByte() {
    return fragment.byteAt(position);
  }

  public NinjaLexerStep nextStep() {
    Preconditions.checkState(error == null);
    Preconditions.checkState(!seenZero);

    return new NinjaLexerStep(fragment, end);
  }

  /**
   * Returns true, if there are still symbols to process, i.e. either the next step can be
   * constructed, or if current step was just created, so its bounds are not known yet.
   */
  public boolean canAdvance() {
    return !seenZero && error == null && end < fragment.length();
  }

  public FileFragment getFragment() {
    return fragment;
  }

  /** Return step bytes, taking into account possible escaped symbol offset. */
  public byte[] getBytes() {
    return fragment.getBytes(position, end);
  }

  public int getPosition() {
    return position;
  }

  public boolean isSeenZero() {
    return seenZero;
  }

  public String getError() {
    return error;
  }

  public int getStart() {
    return position;
  }

  public int getEnd() {
    return end;
  }

  private boolean checkForward(int steps, char... chars) {
    if ((position + steps) < fragment.length()) {
      for (char ch : chars) {
        if ((byte) ch == fragment.byteAt(position + steps)) {
          return true;
        }
      }
    }
    return false;
  }

  public void forceError(String error) {
    this.error = error;
    end = position + 1;
  }

  public void skipSpaces() {
    end = eatSequence(position, aByte -> ' ' != aByte && '\t' != aByte);
  }

  public void skipComment() {
    Preconditions.checkState('#' == fragment.byteAt(position));
    end = eatSequence(position + 1, aByte -> '\n' == aByte || '\r' == aByte);
  }

  public boolean trySkipEscapedNewline() {
    Preconditions.checkState('$' == fragment.byteAt(position));
    if (checkForward(1, '\n')) {
      end = position + 2;
      return true;
    } else if (checkForward(1, '\r')) {
      if (checkForward(2, '\n')) {
        end = position + 3;
      } else {
        error = "Wrong newline separators: \\r should be followed by \\n.";
        end = safeEnd(position + 3);
      }
      return true;
    }
    return false;
  }

  public void processLineFeedNewLine() {
    Preconditions.checkState('\r' == fragment.byteAt(position));
    if (checkForward(1, '\n')) {
      end = position + 2;
    } else {
      error = "Wrong newline separators: \\r should be followed by \\n.";
      end = safeEnd(position + 2);
    }
  }

  public boolean tryReadVariableInBrackets() {
    Preconditions.checkState('$' == fragment.byteAt(position));
    if (checkForward(1, '{')) {
      end = eatSequence(position + 2, aByte -> ' ' != aByte);
      int endOfVariableName = readIdentifier(end, true);
      if (endOfVariableName == end) {
        error = "Variable identifier expected.";
        // Up to the 'wrong' symbol.
        end = endOfVariableName + 1;
      } else {
        end = eatSequence(endOfVariableName, aByte -> ' ' != aByte);
        if (end >= fragment.length() || '}' != fragment.byteAt(end)) {
          error = "Variable end symbol '}' expected.";
          end = safeEnd(end + 1);
        } else {
          ++end;
        }
      }
      return true;
    }
    return false;
  }

  public boolean tryReadSimpleVariable() {
    Preconditions.checkState('$' == fragment.byteAt(position));
    if (position + 1 < fragment.length()
        && IDENTIFIER_SYMBOLS.contains(fragment.byteAt(position + 1))) {
      end = readIdentifier(position + 1, false);
      return true;
    }
    return false;
  }

  public boolean tryReadEscapedLiteral() {
    Preconditions.checkState('$' == fragment.byteAt(position));
    if (checkForward(1, '$', ':', ' ')) {
      // Escaped literal.
      end = position + 2;
      return true;
    }
    return false;
  }

  public void tryReadIdentifier() {
    end = readIdentifier(position, true);
    if (position >= end) {
      error =
          String.format(
              "Symbol '%s' is not allowed in the identifier,"
                  + " the text fragment with the symbol:\n%s\n",
              fragment.subFragment(position, position + 1), fragment.getFragmentAround(position));
      end = position + 1;
    }
  }

  public boolean tryReadDoublePipe() {
    Preconditions.checkState('|' == fragment.byteAt(position));
    if (checkForward(1, '|')) {
      end = position + 2;
      return true;
    }
    return false;
  }

  public void readText() {
    int i = position;
    for (; i < fragment.length(); i++) {
      byte b = fragment.byteAt(i);
      if (0 == b) {
        seenZero = true;
        end = i;
        return;
      }
      if (isTextStopper(b)) {
        break;
      }
    }
    end = i;
  }

  public void readPath() {
    int i = position;
    for (; i < fragment.length(); i++) {
      byte b = fragment.byteAt(i);
      if (0 == b) {
        seenZero = true;
        end = i;
        return;
      }
      if (isPathStopper(b)) {
        break;
      }
    }
    end = i;
  }

  // Optimized, since this is run for each byte in the ninja file. (This has better performance
  // than lookup in a java Set, since TEXT_STOPPERS is small.
  private static boolean isTextStopper(byte b) {
    for (int i = 0; i < TEXT_STOPPERS.length; i++) {
      if (b == TEXT_STOPPERS[i]) {
        return true;
      }
    }
    return false;
  }

  // Optimized, since this is run for each byte in the ninja file. (This has better performance
  // than lookup in a java Set, since PATH_STOPPERS is small.
  private static boolean isPathStopper(byte b) {
    for (int i = 0; i < PATH_STOPPERS.length; i++) {
      if (b == PATH_STOPPERS[i]) {
        return true;
      }
    }
    return false;
  }

  private int readIdentifier(int startFrom, boolean withDot) {
    if (withDot) {
      return eatSequence(startFrom, b -> !IDENTIFIER_SYMBOLS.contains(b) && '.' != b);
    } else {
      return eatSequence(startFrom, b -> !IDENTIFIER_SYMBOLS.contains(b));
    }
  }

  private int safeEnd(int number) {
    return Math.min(fragment.length(), number);
  }

  private int eatSequence(int startFrom, Predicate<Byte> stop) {
    int i = startFrom;
    for (; i < fragment.length(); i++) {
      byte b = fragment.byteAt(i);
      if (0 == b) {
        seenZero = true;
        return i;
      }
      if (stop.test(b)) {
        break;
      }
    }
    return i;
  }

  /**
   * For the quick checks outside of skipXXX and tryXXX methods of this class, assume that the step
   * takes just one symbol.
   */
  public void ensureEnd() {
    if (end < 0) {
      end = position + 1;
    }
  }
}
