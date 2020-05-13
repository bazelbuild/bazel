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
package com.google.devtools.build.lib.profiler;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Range;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.profiler.MetricData.HistogramElement;

/**
 * A stat recorder that can record time histograms, count of calls, average time, Std. Deviation
 * and max time.
 */
@ThreadSafe
public class SingleStatRecorder implements StatRecorder {

  private final int buckets;
  private final Object description;
  private int[] histogram;
  private int count = 0;
  private double avg = 0.0;
  private double m2 = 0.0;
  private int max = -1;

  public SingleStatRecorder(Object description, int buckets) {
    this.description = description;
    Preconditions.checkArgument(buckets > 1, "At least two buckets (one for bellow start and one"
        + "for above start) are required");
    this.buckets = buckets;
    histogram = new int[buckets];
  }

  /** Create an snapshot of the stats recorded up to now. */
  public MetricData snapshot() {
    synchronized (this) {
      ImmutableList.Builder<HistogramElement> result = ImmutableList.builder();
      result.add(new HistogramElement(Range.closedOpen(0, 1), histogram[0]));
      int from = 1;
      for (int i = 1; i < histogram.length - 1; i++) {
        int to = from << 1;
        result.add(new HistogramElement(Range.closedOpen(from, to), histogram[i]));
        from = to;
      }
      result.add(new HistogramElement(Range.atLeast(from), histogram[histogram.length - 1]));
      return new MetricData(description, result.build(), count, avg,
          Math.sqrt(m2 / (double) count), max);
    }
  }

  @Override
  public void addStat(int duration, Object obj) {
    int histogramBucket = Math.min(32 - Integer.numberOfLeadingZeros(duration), buckets - 1);
    synchronized (this) {
      count++;
      double delta = duration - avg;
      avg += delta / count;
      double newDelta = duration - avg;
      m2 += delta * newDelta;
      if (duration > max) {
        max = duration;
      }
      histogram[histogramBucket]++;
    }
  }

  @Override
  public boolean isEmpty() {
    return snapshot().getCount() == 0;
  }

  @Override
  public String toString() {
    return snapshot().toString();
  }
}
