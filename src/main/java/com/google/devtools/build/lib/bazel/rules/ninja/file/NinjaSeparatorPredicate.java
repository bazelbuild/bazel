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
import java.util.ListIterator;
import java.util.Optional;

/**
 * Implementation of {@link SeparatorPredicate} for Ninja files.
 *
 * The Ninja token consists of several text lines;
 * if the line is a part of the previous token, it starts with some amount of spaces or tabs.
 * If the line is the beginning of the new token, it starts with non-space symbol.
 */
public class NinjaSeparatorPredicate implements SeparatorPredicate {
  public static final NinjaSeparatorPredicate INSTANCE = new NinjaSeparatorPredicate();
  private final static byte SPACE_BYTE = (byte) (' ' & 0xff);
  private final static byte TAB_BYTE = (byte) ('\t' & 0xff);

  private NinjaSeparatorPredicate() {
  }

  @Override
  public Optional<Boolean> isSeparator(byte b, ListIterator<Byte> iterator) {
    if ('\n' != b) {
      return Optional.of(false);
    }
    if (!iterator.hasNext()) {
      return Optional.empty();
    }
    byte next = iterator.next();
    iterator.previous();
    return Optional.of(isNotSpace(next));
  }

  private static boolean isNotSpace(byte aByte) {
    return SPACE_BYTE != aByte && TAB_BYTE != aByte;
  }

  @Override
  public boolean splitAdjacent(ListIterator<Byte> leftAtEnd,
      ListIterator<Byte> rightAtBeginning) {
    Preconditions.checkState(!leftAtEnd.hasNext());
    Preconditions.checkState(!rightAtBeginning.hasPrevious());
    if (!leftAtEnd.hasPrevious() || !rightAtBeginning.hasNext()) {
      return false;
    }
    return '\n' == leftAtEnd.previous() && isNotSpace(rightAtBeginning.next());
  }
}
