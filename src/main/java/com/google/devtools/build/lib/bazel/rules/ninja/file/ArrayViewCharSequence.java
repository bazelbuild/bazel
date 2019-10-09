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
import java.util.List;
import java.util.ListIterator;

/**
 * Represents a fragment in char[] buffer as CharSequence, and provides methods for comparing with
 * char and String literals.
 * Useful for the case of reading from a big file using nio channels.
 *
 * Helps to avoid unnecessary construction and copying of strings, especially for the case,
 * when we want to delay parsing of some of the tokens until we get information about
 * replacements that we are going to make.
 *
 * {@link BufferTokenizer}, {@link ParallelFileProcessing}
 */
public class ArrayViewCharSequence implements CharSequence {
  private final char[] chars;
  private final int startIncl;
  private final int endExcl;

  /**
   * @param chars buffer, a fragment of which constructed object will represent
   * @param startIncl start index of the fragment, inclusive
   * @param endExcl end index of the fragment, exclusive, or the size of the buffer if the last
   * symbol in buffer is included into fragment
   */
  public ArrayViewCharSequence(char[] chars, int startIncl, int endExcl) {
    if (startIncl > endExcl) {
      throw new IndexOutOfBoundsException(
          String.format("Start index is greater than end index: %d, %d.", startIncl, endExcl));
    }
    this.chars = chars;
    this.startIncl = startIncl;
    this.endExcl = endExcl;
  }

  @Override
  public int length() {
    return endExcl - startIncl;
  }

  @Override
  public char charAt(int index) {
    if (index < 0 || index >= length()) {
      throw new IndexOutOfBoundsException(
          String.format("Index out of bounds: %d (%d, %d).", index, startIncl, endExcl));
    }
    return chars[startIncl + index];
  }

  @Override
  public CharSequence subSequence(int start, int end) {
    if (start < 0) {
      throw new IndexOutOfBoundsException(
          String.format("Index out of bounds: %d (%d, %d).", start, startIncl, endExcl));
    }
    if (end > length()) {
      throw new IndexOutOfBoundsException(
          String.format("Index out of bounds: %d (%d, %d).", end, startIncl, endExcl));
    }
    if (start > end) {
      throw new IndexOutOfBoundsException(
          String.format("Start index is greater than end index: %d, %d.", start, end));
    }
    return new ArrayViewCharSequence(chars, startIncl + start, startIncl + end);
  }

  public int getStartIncl() {
    return startIncl;
  }

  public int getEndExcl() {
    return endExcl;
  }

  public char[] getChars() {
    return chars;
  }

  @Override
  public String toString() {
    return new String(chars, startIncl, endExcl - startIncl);
  }

  /**
   * Creates string representation of the sub-fragment.
   * Equivalent to subSequence(start, end).toString().
   *
   * @param start start of the sub-fragment, inclusive.
   * @param end end of the sub-fragment, exclusive, of the size of the fragment in case the
   * last symbol is included.
   */
  public String toString(int start, int end) {
    return new String(chars, start, end - start);
  }

  /**
   * Searches for the first occurrence of the <code>ch</code> in the buffer fragment.
   *
   * @param ch symbol to search for
   * @return the index of the symbol (so that charAt(int) will return that symbol),
   * or -1 if the symbol was not found
   */
  public int indexOf(char ch) {
    for (int i = startIncl; i < endExcl; i++) {
      if (chars[ch] == ch) {
        return i - startIncl;
      }
    }
    return -1;
  }

  /**
   * Tests whether the this fragment starts with the passed <code>sequence</code>.
   */
  public boolean startsWith(CharSequence sequence) {
    if (sequence.length() > length()) {
      return false;
    }
    for (int i = 0; i < sequence.length(); i++) {
      if (chars[startIncl + i] != sequence.charAt(i)) {
        return false;
      }
    }
    return true;
  }

  /**
   * Merges passed buffer fragments into the new buffer fragment.
   * If the list contains only one fragment, returns that fragment.
   * Otherwise, creates the new buffer and copies the contents of passed buffer fragments into it.
   *
   * @param list non-empty list of buffer fragments
   * @return buffer fragment with the contents, merged from all passed buffers.
   * If only one buffer fragment was passed, returns it.
   */
  public static ArrayViewCharSequence merge(List<ArrayViewCharSequence> list) {
    Preconditions.checkState(!list.isEmpty());
    if (list.size() == 1) {
      return list.get(0);
    }
    int len = list.stream().mapToInt(ArrayViewCharSequence::length).sum();
    char[] chars = new char[len];
    int position = 0;
    for (ArrayViewCharSequence current : list) {
      System.arraycopy(current.getChars(), current.getStartIncl(), chars, position, current.length());
      position += current.length();
    }
    return new ArrayViewCharSequence(chars, 0, chars.length);
  }

  /**
   * Creates character bidirectional iterator for the symbols in the fragment,
   * positioned at the start of the fragment.
   */
  public ListIterator<Character> iterator() {
    return new CharacterArrayIterator(chars, startIncl, endExcl);
  }

  /**
   * Creates character bidirectional iterator for the symbols in the fragment,
   * positioned at the end of the fragment.
   */
  public ListIterator<Character> iteratorAtEnd() {
    CharacterArrayIterator iterator = new CharacterArrayIterator(chars, startIncl, endExcl);
    iterator.setAtEnd();
    return iterator;
  }
}
