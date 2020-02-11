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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.Iterables;
import java.nio.ByteBuffer;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.List;

/** Represents the fragment of immutable {@link ByteBuffer} for parallel processing. */
public class ByteBufferFragment {
  private final ByteBuffer buffer;
  private final int startIncl;
  private final int endExcl;

  public ByteBufferFragment(ByteBuffer buffer, int startIncl, int endExcl) {
    if (endExcl < startIncl) {
      throw new IndexOutOfBoundsException();
    }
    this.buffer = buffer;
    this.startIncl = startIncl;
    this.endExcl = endExcl;
  }

  @VisibleForTesting
  public int getStartIncl() {
    return startIncl;
  }

  /** Returns the length of fragment. */
  public int length() {
    return endExcl - startIncl;
  }

  /**
   * Creates a sub-fragment of the current fragment.
   *
   * @param from start index, inclusive
   * @param to end index, exclusive, or the size of the buffer, if the last symbol is included
   */
  public ByteBufferFragment subFragment(int from, int to) {
    checkSubBounds(from, to);
    return new ByteBufferFragment(buffer, startIncl + from, startIncl + to);
  }

  public byte byteAt(int index) {
    if (index < 0 || index >= length()) {
      throw new IndexOutOfBoundsException(
          String.format("Index out of bounds: %d (%d, %d).", index, 0, length()));
    }
    return buffer.get(startIncl + index);
  }

  private void checkSubBounds(int from, int to) {
    if (from < 0) {
      throw new IndexOutOfBoundsException(String.format("Index out of bounds: %d.", from));
    }
    if (to > length()) {
      throw new IndexOutOfBoundsException(
          String.format("Index out of bounds: %d (%d, %d).", to, 0, length()));
    }
    if (from > to) {
      throw new IndexOutOfBoundsException(
          String.format("Start index is greater than end index: %d, %d.", from, to));
    }
  }

  public byte[] getBytes(int from, int to) {
    checkSubBounds(from, to);
    int length = to - from;
    byte[] bytes = new byte[length];
    ByteBuffer copy = buffer.duplicate();
    copy.position(startIncl + from);
    copy.get(bytes, 0, length);
    return bytes;
  }

  /**
   * Helper method for forming error messages with text fragment around a place with a problem.
   *
   * @param index position of the problematic symbol.
   * @return a fragment if text around the problematic place, that can be retrieved from this buffer
   */
  public String getFragmentAround(int index) {
    if (index < 0 || index >= length()) {
      throw new IndexOutOfBoundsException(
          String.format("Index out of bounds: %d (%d, %d).", index, 0, length()));
    }
    byte[] bytes = getBytes(Math.max(0, index - 200), Math.min(length(), index + 200));
    return new String(bytes, StandardCharsets.ISO_8859_1);
  }

  private void getBytes(byte[] dst, int offset) {
    if (dst.length - offset < length()) {
      throw new IndexOutOfBoundsException(
          String.format(
              "Not enough space to copy: %d available, %d needed.",
              (dst.length - offset), length()));
    }
    ByteBuffer copy = buffer.duplicate();
    copy.position(startIncl);
    copy.get(dst, offset, (endExcl - startIncl));
  }

  @Override
  public String toString() {
    byte[] bytes = new byte[length()];
    getBytes(bytes, 0);
    return new String(bytes, StandardCharsets.ISO_8859_1);
  }

  /** Decode the underlying bytes using provided charset. */
  public String toString(Charset charset) {
    ByteBuffer duplicate = buffer.duplicate();
    duplicate.limit(endExcl);
    duplicate.position(startIncl);
    return charset.decode(duplicate).toString();
  }

  /**
   * Merges passed buffer fragments into the new buffer fragment. If the list contains only one
   * fragment, returns that fragment. Otherwise, creates the new buffer and copies the contents of
   * passed buffer fragments into it.
   *
   * @param list non-empty list of buffer fragments
   * @return buffer fragment with the contents, merged from all passed buffers. If only one buffer
   *     fragment was passed, returns it.
   */
  @SuppressWarnings("ReferenceEquality")
  public static ByteBufferFragment merge(List<ByteBufferFragment> list) {
    Preconditions.checkState(!list.isEmpty());
    if (list.size() == 1) {
      return list.get(0);
    }
    ByteBuffer first = list.get(0).buffer;
    // We compare contained buffers to be exactly same objects here, i.e. changing to use
    // of equals() is not needed. (Warning suppressed.)
    List<ByteBufferFragment> tail = list.subList(1, list.size());
    if (tail.stream().allMatch(el -> el.buffer == first)) {
      int previousEnd = list.get(0).endExcl;
      for (ByteBufferFragment fragment : tail) {
        Preconditions.checkState(fragment.startIncl == previousEnd);
        previousEnd = fragment.endExcl;
      }
      return new ByteBufferFragment(first, list.get(0).startIncl, Iterables.getLast(list).endExcl);
    }
    int len = list.stream().mapToInt(ByteBufferFragment::length).sum();
    byte[] bytes = new byte[len];
    int position = 0;
    for (ByteBufferFragment current : list) {
      current.getBytes(bytes, position);
      position += current.length();
    }
    return new ByteBufferFragment(ByteBuffer.wrap(bytes), 0, len);
  }
}
