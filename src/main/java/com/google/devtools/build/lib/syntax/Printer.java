// Copyright 2015 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.syntax;

import java.io.IOException;

/**
 * (Pretty) Printing of Skylark values
 */
public final class Printer {

  private Printer() {
  }

  private static Appendable backslashChar(Appendable buffer, char c) throws IOException {
    return buffer.append('\\').append(c);
  }

  private static Appendable escapeCharacter(Appendable buffer, char c, char quote)
      throws IOException {
    if (c == quote) {
      return backslashChar(buffer, c);
    }
    switch (c) {
      case '\\':
        return backslashChar(buffer, '\\');
      case '\r':
        return backslashChar(buffer, 'r');
      case '\n':
        return backslashChar(buffer, 'n');
      case '\t':
        return backslashChar(buffer, 't');
      default:
        if (c < 32) {
          return buffer.append(String.format("\\x%02x", (int) c));
        }
        return buffer.append(c); // no need to support UTF-8
    } // endswitch
  }

  /**
   * Write a properly escaped Skylark representation of a string to a buffer.
   *
   * @param buffer the Appendable we're writing to.
   * @param s the string a representation of which to write.
   * @param quote the quote character to use, '"' or '\''.
   * @return the Appendable, in fluent style.
   */
  public static Appendable writeString(Appendable buffer, String s, char quote)
      throws IOException {
    buffer.append(quote);
    int len = s.length();
    for (int i = 0; i < len; i++) {
      char c = s.charAt(i);
      escapeCharacter(buffer, c, quote);
    }
    return buffer.append(quote);
  }

  /**
   * Write a properly escaped Skylark representation of a string to a buffer.
   * Use standard Skylark convention, i.e., double-quoted single-line string,
   * as opposed to standard Python convention, i.e. single-quoted single-line string.
   *
   * @param buffer the Appendable we're writing to.
   * @param s the string a representation of which to write.
   * @return the buffer, in fluent style.
   */
  public static Appendable writeString(Appendable buffer, String s) throws IOException {
    return writeString(buffer, s, '"');
  }
}
