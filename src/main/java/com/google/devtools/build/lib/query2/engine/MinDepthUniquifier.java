// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.query2.engine;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;

/**
 * A helper for deduping values that have already been seen at certain "depths".
 *
 * <p>This is similar to {@link Uniquifier}.
 */
@ThreadSafe
public interface MinDepthUniquifier<T> {
  /**
   * Returns the subset of {@code newElements} that haven't been seen before at depths less than or
   * equal to {@code depth}
   *
   * <p> There's a natural benign check-then-act race in all concurrent uses of this interface.
   * Imagine we have an element e, two depths d1 and d2 (with d2 < d1), and two threads T1 and T2.
   * T1 may think it's about to be the first one to process e at a depth less than or equal to d1.
   * But before T1 finishes processing e, T2 may think _it's_ about to be first one to process an
   * element at a depth less than or equal to than d2. T1's work is probably wasted.
   */
  ImmutableList<T> uniqueAtDepthLessThanOrEqualTo(Iterable<T> newElements, int depth);
}

