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

package com.google.devtools.build.skyframe;

import com.google.common.collect.ImmutableList;

/** A simple {@link Differencer} that is manually informed of invalid/injected nodes. */
public interface RecordingDifferencer extends Differencer, Injectable {
  @Override
  Diff getDiff(WalkableGraph fromGraph, Version fromVersion, Version toVersion);

  /** Stores the given values for invalidation. */
  void invalidate(Iterable<SkyKey> values);

  /**
   * Invalidates the cached values of any values in error transiently.
   *
   * <p>If a future call to {@link MemoizingEvaluator#evaluate} requests a value that transitively
   * depends on any value that was in an error state (or is one of these), they will be re-computed.
   */
  default void invalidateTransientErrors() {
    // All transient error values have a dependency on the single global ERROR_TRANSIENCE value,
    // so we only have to invalidate that one value to catch everything.
    invalidate(ImmutableList.of(ErrorTransienceValue.KEY));
  }
}
