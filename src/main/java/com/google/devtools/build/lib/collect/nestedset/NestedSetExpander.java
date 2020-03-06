// Copyright 2020 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Stopwatch;
import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.ListenableFuture;
import java.time.Duration;

/**
 * Helper class to expand {@link NestedSet} instances which allows to plug in an extra callbacks for
 * expansions waiting for a {@link ListenableFuture}.
 */
public class NestedSetExpander {
  public static final NestedSetExpander NO_CALLBACKS =
      new NestedSetExpander(
          new ExpansionWithFutureCallbacks() {
            @Override
            public void onSuccessfulExpansion(Duration time, int size) {}

            @Override
            public void onInterruptedExpansion(Duration timeUntilInterrupted) {}
          });

  private final ExpansionWithFutureCallbacks expansionWithFutureCallbacks;

  /** Callbacks invoked if we expand a {@link NestedSet} backed by a {@link ListenableFuture}. */
  public interface ExpansionWithFutureCallbacks {
    void onSuccessfulExpansion(Duration time, int size);

    void onInterruptedExpansion(Duration timeUntilInterrupted);
  }

  public NestedSetExpander(ExpansionWithFutureCallbacks expansionWithFutureCallbacks) {
    this.expansionWithFutureCallbacks = expansionWithFutureCallbacks;
  }

  /**
   * Returns an immutable list of all unique elements of the the provided set, similar to {@link
   * NestedSet#toList()}, but will propagate an {@code InterruptedException} if one is thrown.
   */
  public final <T> ImmutableList<? extends T> toListInterruptibly(NestedSet<? extends T> nestedSet)
      throws InterruptedException {
    if (!(nestedSet.rawChildren() instanceof ListenableFuture)) {
      return nestedSet.toListInterruptibly();
    }

    Stopwatch stopwatch = Stopwatch.createStarted();
    ImmutableList<? extends T> result;
    try {
      result = nestedSet.toListInterruptibly();
    } catch (InterruptedException e) {
      expansionWithFutureCallbacks.onInterruptedExpansion(stopwatch.elapsed());
      throw e;
    }
    expansionWithFutureCallbacks.onSuccessfulExpansion(stopwatch.elapsed(), result.size());
    return result;
  }
}
