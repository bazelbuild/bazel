// Copyright 2016 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.junit.runner.util;

import java.util.Collections;
import java.util.Map;

/**
 * An escaper that uses an array to quickly look up replacement characters for a given {@code char}
 * value. An additional safe range is provided that determines whether {@code char} values without
 * specific replacements are to be considered safe and left unescaped or should be escaped in a
 * general way.
 */
public abstract class CharEscaper {
  // The replacement array.
  private final char[][] replacements;
  // The number of elements in the replacement array.
  private final int replacementsLength;
  // The first character in the safe range.
  private final char safeMin;
  // The last character in the safe range.
  private final char safeMax;

  // The multiplier for padding to use when growing the escape buffer.
  private static final int DEST_PAD_MULTIPLIER = 2;

  public CharEscaper(Map<Character, String> replacementMap, char safeMin, char safeMax) {
    this.replacements = createReplacementArray(replacementMap);
    this.replacementsLength = replacements.length;
    this.safeMin = safeMin;
    this.safeMax = safeMax;
  }

  public final String escape(String s) {
    if (s == null) {
      throw new NullPointerException();
    }
    for (int i = 0; i < s.length(); i++) {
      char c = s.charAt(i);
      if ((c < replacementsLength && replacements[c] != null) || c > safeMax || c < safeMin) {
        return escapeSlow(s, i);
      }
    }
    return s;
  }

  /**
   * A thread-local destination buffer to keep us from creating new buffers. The starting size is
   * 1024 characters.
   */
  private static final ThreadLocal<char[]> DEST_TL =
      new ThreadLocal<char[]>() {
        @Override
        protected char[] initialValue() {
          return new char[1024];
        }
      };

  /**
   * Returns the escaped form of a given literal string, starting at the given index. This method is
   * called by the {@link #escape(String)} method when it discovers that escaping is required.
   *
   * @param s the literal string to be escaped
   * @param index the index to start escaping from
   * @return the escaped form of {@code string}
   * @throws NullPointerException if {@code string} is null
   */
  final String escapeSlow(String s, int index) {
    int slen = s.length();

    // Get a destination buffer and setup some loop variables.
    char[] dest = DEST_TL.get();
    int destSize = dest.length;
    int destIndex = 0;
    int lastEscape = 0;

    // Loop through the rest of the string, replacing when needed into the
    // destination buffer, which gets grown as needed as well.
    for (; index < slen; index++) {

      // Get a replacement for the current character.
      char[] r = escape(s.charAt(index));

      // If no replacement is needed, just continue.
      if (r == null) {
        continue;
      }

      int rlen = r.length;
      int charsSkipped = index - lastEscape;

      // This is the size needed to add the replacement, not the full size
      // needed by the string. We only regrow when we absolutely must, and
      // when we do grow, grow enough to avoid excessive growing. Grow.
      int sizeNeeded = destIndex + charsSkipped + rlen;
      if (destSize < sizeNeeded) {
        destSize = sizeNeeded + DEST_PAD_MULTIPLIER * (slen - index);
        dest = growBuffer(dest, destIndex, destSize);
      }

      // If we have skipped any characters, we need to copy them now.
      if (charsSkipped > 0) {
        s.getChars(lastEscape, index, dest, destIndex);
        destIndex += charsSkipped;
      }

      // Copy the replacement string into the dest buffer as needed.
      if (rlen > 0) {
        System.arraycopy(r, 0, dest, destIndex, rlen);
        destIndex += rlen;
      }
      lastEscape = index + 1;
    }

    // Copy leftover characters if there are any.
    int charsLeft = slen - lastEscape;
    if (charsLeft > 0) {
      int sizeNeeded = destIndex + charsLeft;
      if (destSize < sizeNeeded) {

        // Regrow and copy, expensive! No padding as this is the final copy.
        dest = growBuffer(dest, destIndex, sizeNeeded);
      }
      s.getChars(lastEscape, slen, dest, destIndex);
      destIndex = sizeNeeded;
    }
    return new String(dest, 0, destIndex);
  }

  final char[] escape(char c) {
    if (c < replacementsLength) {
      char[] chars = replacements[c];
      if (chars != null) {
        return chars;
      }
    }
    if (c >= safeMin && c <= safeMax) {
      return null;
    }
    return escapeUnsafe(c);
  }

  abstract char[] escapeUnsafe(char c);

  /**
   * Helper method to grow the character buffer as needed, this only happens once in a while so it's
   * ok if it's in a method call. If the index passed in is 0 then no copying will be done.
   */
  private static char[] growBuffer(char[] dest, int index, int size) {
    char[] copy = new char[size];
    if (index > 0) {
      System.arraycopy(dest, 0, copy, 0, index);
    }
    return copy;
  }

  private static char[][] createReplacementArray(Map<Character, String> map) {
    if (map == null) {
      throw new NullPointerException();
    }
    if (map.isEmpty()) {
      return new char[0][0];
    }
    char max = Collections.max(map.keySet());
    char[][] replacements = new char[max + 1][];
    for (char c : map.keySet()) {
      replacements[c] = map.get(c).toCharArray();
    }
    return replacements;
  }
}

