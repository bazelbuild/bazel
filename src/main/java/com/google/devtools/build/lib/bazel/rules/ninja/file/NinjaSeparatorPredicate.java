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

import java.util.function.BiPredicate;

/**
 * Implementation of separator predicate for Ninja files.
 *
 * The Ninja token consists of several text lines;
 * if the line is a part of the previous token, it starts with some amount of spaces or tabs.
 * If the line is the beginning of the new token, it starts with non-space symbol.
 */
public class NinjaSeparatorPredicate implements BiPredicate<Byte, Byte> {
  public static final NinjaSeparatorPredicate INSTANCE = new NinjaSeparatorPredicate();
  private final static byte SPACE_BYTE = (byte) (' ' & 0xff);
  private final static byte TAB_BYTE = (byte) ('\t' & 0xff);

  private NinjaSeparatorPredicate() {
  }

  @Override
  public boolean test(Byte current, Byte next) {
    return '\n' == current && isNotSpace(next);
  }

  private static boolean isNotSpace(byte aByte) {
    return SPACE_BYTE != aByte && TAB_BYTE != aByte;
  }
}
