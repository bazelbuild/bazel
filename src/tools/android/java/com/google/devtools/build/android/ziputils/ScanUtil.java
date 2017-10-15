// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.ziputils;

import com.google.common.base.Preconditions;
import java.nio.ByteBuffer;

/**
 * Helper class for finding byte patterns in a byte buffer, scanning forwards or backwards.
 * This is used to located the "end of central directory" marker, and in other instances
 * where zip file elements must be located by scanning for known signatures.
 * For instance, when it's necessary to scan for zip entries, without
 * relying on the central directory to provide the exact location, or when scanning for
 * data descriptor at the end of an entry of unknown size.
 */
public class ScanUtil {

  /**
   * Finds the next previous position of a given byte sequence in a given byte buffer,
   * starting from the buffer's current position (excluded).
   *
   * @param target byte sequence to search for.
   * @param buffer byte buffer in which to search.
   * @return position of the last location of the target sequence, at a position
   * strictly smaller than the current position, or -1 if not found.
   * @throws IllegalArgumentException if either {@code target} or {@code buffer} is {@code null}.
   */
  static int scanBackwardsTo(byte[] target, ByteBuffer buffer) {
    Preconditions.checkNotNull(target);
    Preconditions.checkNotNull(buffer);
    if (target.length == 0) {
      return buffer.position() - 1;
    }
    int pos = buffer.position() - target.length;
    if (pos < 0) {
      return -1;
    }
    scan:
    while (true) {
      while (pos >= 0 && buffer.get(pos) != target[0]) {
        pos--;
      }
      if (pos < 0) {
        return -1;
      }
      for (int i = 1; i < target.length; i++) {
        if (buffer.get(pos + i) != target[i]) {
          pos--;
          continue scan;
        }
      }
      return pos;
    }
  }

  /**
   * Finds the next position of a given byte sequence in a given byte buffer, starting from the
   * buffer's current position (included).
   *
   * @param target byte sequence to search for.
   * @param buffer byte buffer in which to search.
   * @return position of the first location of the target sequence, or -1 if not found.
   * @throws IllegalArgumentException if either {@code target} or {@code buffer} is {@code null}.
   */
  static int scanTo(byte[] target, ByteBuffer buffer) {
    Preconditions.checkNotNull(target);
    Preconditions.checkNotNull(buffer);
    if (!buffer.hasRemaining()) {
      return -1;
    }
    int pos = buffer.position();
    if (target.length == 0) {
      return pos;
    }
    scan:
    while (true) {
      while (pos <= buffer.limit() - target.length && buffer.get(pos) != target[0]) {
        pos++;
      }
      if (pos > buffer.limit() - target.length + 1) {
        return -1;
      }
      for (int i = 1; i < target.length; i++) {
        if (buffer.get(pos + i) != target[i]) {
          pos++;
          continue scan;
        }
      }
      return pos;
    }
  }
}
