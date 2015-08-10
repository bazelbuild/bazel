// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.devtools.build.lib.concurrent.ThreadSafety;

import javax.annotation.Nullable;

/**
 * Receiver to inform callers which values have been invalidated. Values may be invalidated and then
 * re-validated if they have been found not to be changed.
 */
public interface EvaluationProgressReceiver {
  /**
   * New state of the value entry after evaluation.
   */
  enum EvaluationState {
    /** The value was successfully re-evaluated. */
    BUILT,
    /** The value is clean or re-validated. */
    CLEAN,
  }

  /**
   * New state of the value entry after invalidation.
   */
  enum InvalidationState {
    /** The value is dirty, although it might get re-validated again. */
    DIRTY,
    /** The value is dirty and got deleted, cannot get re-validated again. */
    DELETED,
  }

  /**
   * Notifies that the node named by {@code key} has been invalidated.
   *
   * <p>{@code state} indicates the new state of the value.
   *
   * <p>May be called concurrently from multiple threads, possibly with the same {@code key}.
   */
  @ThreadSafety.ThreadSafe
  void invalidated(SkyKey skyKey, InvalidationState state);

  /**
   * Notifies that {@code skyKey} is about to get queued for evaluation.
   *
   * <p>Note that we don't guarantee that it actually got enqueued or will, only that if
   * everything "goes well" (e.g. no interrupts happen) it will.
   *
   * <p>This guarantee is intentionally vague to encourage writing robust implementations.
   */
  @ThreadSafety.ThreadSafe
  void enqueueing(SkyKey skyKey);

  /**
   * Notifies that the node for {@code skyKey} has been evaluated.
   *
   * <p>{@code state} indicates the new state of the node.
   *
   * <p>If the value builder threw an error when building this node, then {@code value} is null.
   */
  @ThreadSafety.ThreadSafe
  void evaluated(SkyKey skyKey, @Nullable SkyValue value, EvaluationState state);
}
