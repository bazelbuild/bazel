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

import java.time.Duration;

/**
 * Converts a set of ranges into a graph by counting the number of ranges that are active at any
 * point in time. Time is split into equal-sized buckets, and we compute one value per bucket. If a
 * range partially overlaps a bucket, then the bucket is incremented by the fraction of overlap.
 */
public interface TimeSeries {

  /** Adds a new range to the time series, by increasing every affected bucket by 1. */
  void addRange(Duration startTime, Duration endTime);

  /** Adds a new range to the time series, by increasing every affected bucket by value. */
  void addRange(Duration rangeStart, Duration rangeEnd, double value);

  double[] toDoubleArray(int len);
}
