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

import java.util.HashMap;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Utility class for dealing with escaping XML content and attributes.
 */
public class XmlEscapers {
  private XmlEscapers() {}

  private static final char MIN_ASCII_CONTROL_CHAR = 0x00;
  private static final char MAX_ASCII_CONTROL_CHAR = 0x1F;

  public static CharEscaper xmlContentEscaper() {
    return XML_CONTENT_ESCAPER;
  }

  public static CharEscaper xmlAttributeEscaper() {
    return XML_ATTRIBUTE_ESCAPER;
  }

  private static final CharEscaper XML_CONTENT_ESCAPER;
  private static final CharEscaper XML_ATTRIBUTE_ESCAPER;

  static {
    Builder builder = Builder.builder();
    builder.setSafeRange(Character.MIN_VALUE, '\uFFFD');
    builder.setUnsafeReplacement("\uFFFD");

    for (char c = MIN_ASCII_CONTROL_CHAR; c <= MAX_ASCII_CONTROL_CHAR; c++) {
      if (c != '\t' && c != '\n' && c != '\r') {
        builder.addEscape(c, "\uFFFD");
      }
    }

    builder.addEscape('&', "&amp;");
    builder.addEscape('<', "&lt;");
    builder.addEscape('>', "&gt;");
    XML_CONTENT_ESCAPER = builder.build();
    builder.addEscape('\'', "&apos;");
    builder.addEscape('"', "&quot;");
    builder.addEscape('\t', "&#x9;");
    builder.addEscape('\n', "&#xA;");
    builder.addEscape('\r', "&#xD;");
    XML_ATTRIBUTE_ESCAPER = builder.build();
  }

  /**
   * A builder for CharEscaper.
   */
  static final class Builder {
    private final Map<Character, String> replacementMap = new HashMap<>();
    private char safeMin = Character.MIN_VALUE;
    private char safeMax = Character.MAX_VALUE;
    private String unsafeReplacement = null;

    static Builder builder() {
      return new Builder();
    }
    // The constructor is exposed via the builder() method above.
    private Builder() {}

    /**
     * Sets the safe range of characters for the escaper. Characters in this range that have no
     * explicit replacement are considered 'safe' and remain unescaped in the output. If
     * {@code safeMax < safeMin} then the safe range is empty.
     *
     * @return the builder instance
     */
    Builder setSafeRange(char safeMin, char safeMax) {
      this.safeMin = safeMin;
      this.safeMax = safeMax;
      return this;
    }

    /**
     * Sets the replacement string for any characters outside the 'safe' range that have no explicit
     * replacement. If {@code unsafeReplacement} is {@code null} then no replacement will occur, if
     * it is {@code ""} then the unsafe characters are removed from the output.
     *
     * @return the builder instance
     */
    Builder setUnsafeReplacement(@Nullable String unsafeReplacement) {
      this.unsafeReplacement = unsafeReplacement;
      return this;
    }

    /**
     * Adds a replacement string for the given input character. The specified character will be
     * replaced by the given string whenever it occurs in the input, irrespective of whether it lies
     * inside or outside the 'safe' range.
     *
     * @return the builder instance
     * @throws NullPointerException if {@code replacement} is null
     */
    Builder addEscape(char c, String replacement) {
      if (replacement == null) {
        throw new NullPointerException();
      }
      // This can replace an existing character (the builder is re-usable).
      replacementMap.put(c, replacement);
      return this;
    }

    /**
     * Returns a new CharEscaper based on the current state of the builder.
     */
    CharEscaper build() {
      return new CharEscaper(replacementMap, safeMin, safeMax) {
        private final char[] replacementChars =
            unsafeReplacement != null ? unsafeReplacement.toCharArray() : null;

        @Override
        char[] escapeUnsafe(char c) {
          return replacementChars;
        }
      };
    }
  }
}

