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

package com.google.devtools.build.lib.bazel.rules.ninja.file;

import com.google.common.base.Preconditions;

/**
 * Finds the border between two different Ninja declarations.
 *
 * <p>The Ninja declaration consists of several text lines; if the line is a part of the previous
 * declaration, it starts with some amount of spaces or tabs. If the line is the beginning of the
 * new declaration, it starts with non-space symbol. Dollar symbol '$' escapes the newline, i.e.
 * "$\nsomething" does not contain a separator.
 *
 * <p>We support '\r\n' separators in Ninja files and throw {@link IncorrectSeparatorException} in
 * case an incorrect separator '\r' is used.
 */
public class NinjaSeparatorFinder {
  private static final byte DOLLAR_BYTE = '$';
  private static final byte LINEFEED_BYTE = '\r';
  private static final byte NEWLINE_BYTE = '\n';
  private static final byte SPACE_BYTE = ' ';
  private static final byte TAB_BYTE = '\t';

  private NinjaSeparatorFinder() {}

  public static int findNextSeparator(FileFragment fragment, int startingFrom, int untilExcluded)
      throws IncorrectSeparatorException {
    Preconditions.checkState(startingFrom < fragment.length());
    Preconditions.checkState(untilExcluded < 0 || untilExcluded <= fragment.length());

    boolean escaped = DOLLAR_BYTE == fragment.byteAt(startingFrom);
    int endExcl = untilExcluded > 0 ? untilExcluded : fragment.length();
    for (int i = startingFrom + 1; i < endExcl - 1; i++) {
      byte current = fragment.byteAt(i);
      byte next = fragment.byteAt(i + 1);
      byte afterNextOrSpace = i < (endExcl - 2) ? fragment.byteAt(i + 2) : SPACE_BYTE;
      if (LINEFEED_BYTE == current && NEWLINE_BYTE != next) {
        throw new IncorrectSeparatorException(
            "Wrong newline separators: \\r should be followed by \\n.");
      }
      if (!escaped
          && SPACE_BYTE != afterNextOrSpace
          && TAB_BYTE != afterNextOrSpace
          && LINEFEED_BYTE == current) {
        // To do not introduce the length of the separator, let us point to the last symbol of it.
        return i + 1;
      }
      if (!escaped && SPACE_BYTE != next && TAB_BYTE != next && NEWLINE_BYTE == current) {
        return i;
      }
      if (escaped && LINEFEED_BYTE == current) {
        // Jump over the whole escaped linefeed + newline.
        ++i;
        escaped = false;
      } else {
        escaped = DOLLAR_BYTE == current;
      }
    }
    return -1;
  }
}
