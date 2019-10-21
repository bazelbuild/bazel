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

/**
 * Implementation of {@link SeparatorPredicate} for Ninja files.
 *
 * <p>The Ninja declaration consists of several text lines; if the line is a part of the previous
 * declaration, it starts with some amount of spaces or tabs. If the line is the beginning of the
 * new declaration, it starts with non-space symbol. Dollar symbol '$' escapes the newline, i.e.
 * "$\nsomething" does not contain a separator.
 */
public class NinjaSeparatorPredicate implements SeparatorPredicate {
  public static final NinjaSeparatorPredicate INSTANCE = new NinjaSeparatorPredicate();

  private static final byte DOLLAR_BYTE = '$';
  private static final byte NEWLINE_BYTE = '\n';
  private static final byte SPACE_BYTE = ' ';
  private static final byte TAB_BYTE = '\t';

  private NinjaSeparatorPredicate() {}

  @Override
  public boolean test(byte previous, byte current, byte next) {
    return NEWLINE_BYTE == current
        && DOLLAR_BYTE != previous
        && SPACE_BYTE != next
        && TAB_BYTE != next;
  }
}
