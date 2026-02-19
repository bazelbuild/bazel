// Copyright 2026 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.util;

import com.google.common.collect.ImmutableList;
import java.util.ArrayList;

/**
 * A class that buckets values into buckets based on the length of the decimal representation of the
 * value and its leading digit.
 *
 * <p>To use this class, call {@link #add} to add values to the buckets. Then call {@link
 * #getBuckets} to get the buckets.
 */
public class DecimalBucketer {
  /** A bucket of values. min is inclusive, max is exclusive. */
  public record Bucket(long minInclusive, long maxExclusive, long count) {}

  private final ArrayList<Long> counts = new ArrayList<>();

  public DecimalBucketer() {}

  /** Adds a value to the bucketer. It must be non-negative. */
  public synchronized void add(long value) {
    if (value < 0) {
      throw new IllegalArgumentException("value must be non-negative");
    }

    // Each length has 9 buckets, one for each leading digit, except for 0-9, which has 10.
    int bucketIdx = 0;
    while (value >= 10) {
      value /= 10;
      bucketIdx += 9;
    }
    bucketIdx += (int) value; // value here is always >0 except if the input is 0 so this works out

    while (counts.size() <= bucketIdx) {
      counts.add(0L);
    }
    counts.set(bucketIdx, counts.get(bucketIdx) + 1L);
  }

  /** Returns the buckets in which there are values in increasing order of the bucket minimum. */
  public synchronized ImmutableList<Bucket> getBuckets() {
    ImmutableList.Builder<Bucket> builder = ImmutableList.builder();

    long base = 1;
    long leadingDigit = 0;

    for (long count : counts) {
      if (count > 0) {
        long min = base * leadingDigit;
        long max = Long.MAX_VALUE - base < min ? Long.MAX_VALUE : min + base;
        builder.add(new Bucket(min, max, count));
      }

      leadingDigit += 1;
      if (leadingDigit > 9) {
        leadingDigit = 1;
        base *= 10;
      }
    }
    return builder.build();
  }
}
