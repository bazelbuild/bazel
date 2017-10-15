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

import static java.util.stream.Collectors.joining;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Range;
import java.text.DecimalFormat;

/**
 * Metric data for {@code description} object. Contains count, average, standard deviation, max and
 * histogram.
 */
public final class MetricData {

  private final Object description;
  private final ImmutableList<HistogramElement> histogram;
  private final int count;
  private final double avg;
  private final double stdDev;
  private final int max;
  public MetricData(Object description, ImmutableList<HistogramElement> histogram, int count,
      double avg, double stdDev, int max) {
    this.description = description;
    this.histogram = histogram;
    this.count = count;
    this.avg = avg;
    this.stdDev = stdDev;
    this.max = max;
  }

  public Object getDescription() {
    return description;
  }

  public ImmutableList<HistogramElement> getHistogram() {
    return histogram;
  }

  public int getCount() {
    return count;
  }

  public double getAvg() {
    return avg;
  }

  public double getStdDev() {
    return stdDev;
  }

  public int getMax() {
    return max;
  }

  @Override
  public String toString() {
    if (count == 0) {
      return "'" + description + "'. Zero data recorded";
    }
    DecimalFormat fmt = new DecimalFormat("0.###");
    return "'"
        + description
        + "'. "
        + " Count: "
        + count
        + " Avg: "
        + fmt.format(avg)
        + " StdDev: "
        + fmt.format(stdDev)
        + " Max: "
        + max
        + " Histogram:\n  "
        + histogram
            .stream()
            .filter(element -> element.count > 0)
            .map(Object::toString)
            .collect(joining("\n  "));
  }

  /** An histogram element that contains the range that applies to and the number of elements. */
  public static final class HistogramElement {

    private final Range<Integer> range;
    private final int count;

    HistogramElement(Range<Integer> range, int count) {
      this.range = range;
      this.count = count;
    }

    public Range<Integer> getRange() {
      return range;
    }

    public int getCount() {
      return count;
    }

    @Override
    public String toString() {

      return String.format("%-15s:%10s",
          "[" + range.lowerEndpoint() + ".." + (range.hasUpperBound()
                                                ? range.upperEndpoint()
                                                : "\u221e") // infinite symbol
              + " ms]", count);
    }
  }
}
