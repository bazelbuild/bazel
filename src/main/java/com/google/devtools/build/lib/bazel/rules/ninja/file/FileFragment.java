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
public class FileFragment {
  /** The backing {@link ByteBuffer}. May be shared with other {@link FileFragment} instances. */
  private final ByteBuffer buffer;

  /** The offset in the file the backing {@link ByteBuffer} starts at. */
  private final long fileOffset;

  /**
   * The start of this fragment in the backing {@link ByteBuffer}.
   *
   * <p>The file offset of the fragment is the sum of this value and the offset of the backing
   * buffer ({@code fileOffset}).
   */
  private final int startIncl;

  /** The end of this fragment in the backing {@link ByteBuffer}. */
  private final int endExcl;

  public FileFragment(ByteBuffer buffer, long fileOffset, int startIncl, int endExcl) {
    if (endExcl < startIncl) {
      throw new IndexOutOfBoundsException();
    }
    this.buffer = buffer;
    this.fileOffset = fileOffset;
    this.startIncl = startIncl;
    this.endExcl = endExcl;
  }

  /**
   * Returns the start of this fragment in the backing {@link ByteBuffer}, that is, relative to the
   * {@code fileOffset} in the file.
   */
  @VisibleForTesting
  public int getStartIncl() {
    return startIncl;
  }

  /** The byte offset in the file the backing {@link ByteBuffer} starts at. */
  public long getFileOffset() {
    return fileOffset;
  }

  /** The offset in the file this fragment starts at. */
  public long getFragmentOffset() {
    return fileOffset + startIncl;
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
  public FileFragment subFragment(int from, int to) {
    checkSubBounds(from, to);
    return new FileFragment(buffer, fileOffset, startIncl + from, startIncl + to);
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

  public byte[] getBytes(int from, int to) {
    checkSubBounds(from, to);
    int length = to - from;
    byte[] bytes = new byte[length];
    ByteBuffer copy = buffer.duplicate();
    copy.position(startIncl + from);
    copy.get(bytes, 0, length);
    return bytes;
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
   * Merges multiple file fragments into a single one.
   *
   * <p>If needed, also creates a new backing {@link ByteBuffer} (if the fragments on the input were
   * not contiguous in a single buffer)
   *
   * @param list non-empty list of file fragments
   * @return the merged file fragment
   */
  @SuppressWarnings("ReferenceEquality")
  public static FileFragment merge(List<FileFragment> list) {
    Preconditions.checkState(!list.isEmpty());
    if (list.size() == 1) {
      return list.get(0);
    }
    ByteBuffer first = list.get(0).buffer;
    long firstOffset = list.get(0).fileOffset;

    // We compare contained fragments to be exactly same objects here, i.e. changing to use
    // of equals() is not needed. (Warning suppressed.)
    List<FileFragment> tail = list.subList(1, list.size());
    if (tail.stream().allMatch(el -> el.buffer == first)) {
      int previousEnd = list.get(0).endExcl;
      for (FileFragment fragment : tail) {
        Preconditions.checkState(fragment.startIncl == previousEnd);
        previousEnd = fragment.endExcl;
      }
      return new FileFragment(
          first, firstOffset, list.get(0).startIncl, Iterables.getLast(list).endExcl);
    }
    int len = list.stream().mapToInt(FileFragment::length).sum();
    byte[] bytes = new byte[len];
    int position = 0;
    for (FileFragment current : list) {
      current.getBytes(bytes, position);
      position += current.length();
    }
    return new FileFragment(
        ByteBuffer.wrap(bytes), firstOffset + list.get(0).getStartIncl(), 0, len);
  }
}
