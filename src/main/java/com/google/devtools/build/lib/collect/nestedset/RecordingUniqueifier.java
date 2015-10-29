// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.collect.nestedset;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.collect.CompactHashSet;

import java.util.BitSet;
import java.util.Set;

/**
 * A Uniqueifier that records a transcript of its interactions with the underlying Set.  A memo can
 * then be retrieved in the form of another Uniqueifier, and given the same sequence of isUnique
 * calls, the Set interactions can be avoided.
 */
class RecordingUniqueifier implements Uniqueifier {
  /**
   * Unshared byte memos under this length are not constructed.
   */
  @VisibleForTesting static final int LENGTH_THRESHOLD = 4096 / 8; // bits -> bytes

  /**
   * Returned as a marker that memoization was not worthwhile.
   */
  private static final Object NO_MEMO = new Object();

  /**
   * Shared one-byte memos.
   */
  private static final byte[][] SHARED_SMALL_MEMOS_1;
  
  /**
   * Shared two-byte memos.
   */
  private static final byte[][] SHARED_SMALL_MEMOS_2;

  static {
    // Create interned arrays for one and two byte memos
    // The memos always start with 0x3, so some one and two byte arrays can be skipped.

    byte[][] memos1 = new byte[64][1];
    for (int i = 0; i < 64; i++) {
      memos1[i][0] = (byte) ((i << 2) | 0x3);
    }
    SHARED_SMALL_MEMOS_1 = memos1;

    byte[][] memos2 = new byte[16384][2];
    for (int i = 0; i < 64; i++) {
      byte iAdj = (byte) (0x3 | (i << 2));
      for (int j = 0; j < 256; j++) {
        int idx = i | (j << 6);
        memos2[idx][0] = iAdj;
        memos2[idx][1] = (byte) j;
      }
    }
    SHARED_SMALL_MEMOS_2 = memos2;
  }

  private final Set<Object> witnessed;
  private final BitSet memo = new BitSet();
  private int idx = 0;

  RecordingUniqueifier() {
    this(256);
  }

  RecordingUniqueifier(int expectedSize) {
    this.witnessed = CompactHashSet.createWithExpectedSize(expectedSize);
  }

  static Uniqueifier createReplayUniqueifier(Object memo) {
    if (memo == NO_MEMO) {
      // We receive NO_MEMO for nested sets that are under a size threshold. Therefore, use a
      // smaller initial size than the default.
      return new RecordingUniqueifier(64);
    } else if (memo instanceof Integer) {
      BitSet bs = new BitSet();
      bs.set(0, (Integer) memo);
      return new ReplayUniqueifier(bs);
    }
    return new ReplayUniqueifier(BitSet.valueOf((byte[]) memo));
  }

  @Override
  public boolean isUnique(Object o) {
    boolean firstInstance = witnessed.add(o);
    memo.set(idx++, firstInstance);
    return firstInstance;
  }

  /**
   * Gets the memo of the set interactions.  Do not call isUnique after this point.
   */
  Object getMemo() {
    this.idx = -1; // will cause failures if isUnique is called after getMemo.

    // If the bitset is just a contiguous block of ones, use a length memo
    int length = memo.length();
    if (memo.cardinality() == length) {
      return length; // length-based memo
    }

    byte[] ba = memo.toByteArray();

    Preconditions.checkState(
        (length < 2) || ((ba[0] & 3) == 3),
        "The memo machinery expects memos to always begin with two 1 bits, "
            + "but instead, this memo starts with %s.", ba[0]);
    
    // For short memos, use an interned array for the memo
    if (ba.length == 1) {
      return SHARED_SMALL_MEMOS_1[(0xFF & ba[0]) >>> 2]; // shared memo
    } else if (ba.length == 2) {
      return SHARED_SMALL_MEMOS_2[((ba[1] & 0xFF) << 6) | ((0xFF & ba[0]) >>> 2)];
    }

    // For mid-sized cases, skip the memo since it is not worthwhile
    if (ba.length < LENGTH_THRESHOLD) {
      return NO_MEMO; // skipped memo
    }

    return ba; // normal memo
  }

  private static final class ReplayUniqueifier implements Uniqueifier {
    private final BitSet memo;
    private int idx = 0;

    ReplayUniqueifier(BitSet memo) {
      this.memo = memo;
    }

    @Override
    public boolean isUnique(Object o) {
      return memo.get(idx++);
    }
  }
}
