// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.profiler;

import static java.lang.Math.max;

import java.time.Duration;
import java.util.Arrays;
import javax.annotation.concurrent.GuardedBy;

/**
 * Converts a set of ranges into a graph by counting the number of ranges that are active at any
 * point in time. Time is split into equal-sized buckets, and we compute one value per bucket. If a
 * range partially overlaps a bucket, then the bucket is incremented by the fraction of overlap.
 */
public class TimeSeries {
  private final Duration startTime;
  private final long bucketSizeMillis;
  private static final int INITIAL_SIZE = 100;

  @GuardedBy("this")
  private double[] data = new double[INITIAL_SIZE];

  public TimeSeries(Duration startTime, Duration bucketDuration) {
    this.startTime = startTime;
    this.bucketSizeMillis = bucketDuration.toMillis();
  }

  public void addRange(Duration startTime, Duration endTime) {
    addRange(startTime, endTime, /* value= */ 1);
  }

  /** Adds a new range to the time series, by increasing every affected bucket by value. */
  public void addRange(Duration rangeStart, Duration rangeEnd, double value) {
    // Compute times relative to start and their positions in the data array.
    rangeStart = rangeStart.minus(startTime);
    rangeEnd = rangeEnd.minus(startTime);
    int startPosition = (int) (rangeStart.toMillis() / bucketSizeMillis);
    int endPosition = (int) (rangeEnd.toMillis() / bucketSizeMillis);

    // Assume we add the following range R:
    // ----------------------------------
    // |     |ssRRR|RRRRR|Reeee|      |
    // ----------------------------------
    // we cannot just add value to each affected bucket but have to correct the values for the first
    // and last bucket by calculating the size of 's' and 'e'.
    double missingStartFraction =
        ((double) rangeStart.minusMillis(bucketSizeMillis * startPosition).toMillis())
            / bucketSizeMillis;
    double missingEndFraction =
        ((double) (bucketSizeMillis * (endPosition + 1) - rangeEnd.toMillis())) / bucketSizeMillis;

    if (startPosition < 0) {
      startPosition = 0;
      missingStartFraction = 0;
    }
    if (endPosition < startPosition) {
      endPosition = startPosition;
      missingEndFraction = 0;
    }

    synchronized (this) {
      // Resize data array if necessary so it can at least fit endPosition.
      if (endPosition >= data.length) {
        data = Arrays.copyOf(data, max(endPosition + 1, 2 * data.length));
      }

      // Do the actual update.
      for (int i = startPosition; i <= endPosition; i++) {
        double fraction = 1;
        if (i == startPosition) {
          fraction -= missingStartFraction;
        }
        if (i == endPosition) {
          fraction -= missingEndFraction;
        }
        data[i] += fraction * value;
      }
    }
  }

  public synchronized double[] toDoubleArray(int len) {
    return Arrays.copyOf(data, len);
  }
}
