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

import java.util.ListIterator;
import java.util.NoSuchElementException;

/**
 * Immutable bidirectional iterator for the fragment of the character buffer.
 * Intended to be used together with {@link CharSequence} implementation of the character array
 * fragment: {@link ArrayViewCharSequence}.
 */
public class CharacterArrayIterator implements ListIterator<Character> {
  private final char[] chars;
  private int index;
  private final int startIncl;
  private final int endIndexExcl;

  /**
   * @param chars character buffer, fragment of which is iterable with constructed object
   * @param startIncl start index in the <code>chars</code> buffer, inclusive
   * @param endExcl end index in the <code>chars</code> buffer, exclusive, or the size
   * of the buffer, if the last element is included.
   */
  public CharacterArrayIterator(char[] chars, int startIncl, int endExcl) {
    if (startIncl > endExcl) {
      throw new IndexOutOfBoundsException(
          String.format("Start index is greater than end index: %d, %d.", startIncl, endExcl));
    }
    this.chars = chars;
    this.index = startIncl;
    this.startIncl = startIncl;
    this.endIndexExcl = endExcl;
  }

  /**
   * Positions the iterator at end of the fragment, so that consequent calls return:
   * hasNext() - false, hasPrevious() - true.
   */
  void setAtEnd() {
    index = endIndexExcl;
  }

  @Override
  public boolean hasNext() {
    return index < endIndexExcl;
  }

  @Override
  public Character next() {
    if (index < startIncl || index >= endIndexExcl) {
      throw new NoSuchElementException();
    }
    return chars[index++];
  }

  @Override
  public boolean hasPrevious() {
    return index > startIncl;
  }

  @Override
  public Character previous() {
    if ((index - 1) < startIncl || (index - 1) >= endIndexExcl) {
      throw new NoSuchElementException();
    }
    return chars[--index];
  }

  @Override
  public int nextIndex() {
    return index - startIncl;
  }

  @Override
  public int previousIndex() {
    return index - 1 - startIncl;
  }

  @Override
  public void remove() {
    throw new UnsupportedOperationException();
  }

  @Override
  public void set(Character character) {
    throw new UnsupportedOperationException();
  }

  @Override
  public void add(Character character) {
    throw new UnsupportedOperationException();
  }
}
