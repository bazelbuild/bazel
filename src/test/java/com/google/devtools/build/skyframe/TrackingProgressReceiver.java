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
package com.google.devtools.build.skyframe;

import com.google.common.base.Preconditions;
import com.google.common.collect.Sets;
import com.google.devtools.build.skyframe.NodeEntry.DirtyType;
import java.util.Set;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.Nullable;

/** A testing utility to keep track of evaluation. */
public class TrackingProgressReceiver implements EvaluationProgressReceiver {
  private final boolean checkEvaluationResults;

  /**
   * Callback to be executed on a next {@link #dirtied} or {@link #deleted(SkyKey)} call. It will be
   * run once and is expected to be run if set.
   */
  private final AtomicReference<Runnable> nextInvalidationCallback = new AtomicReference<>();

  public final Set<SkyKey> dirty = Sets.newConcurrentHashSet();
  public final Set<SkyKey> deleted = Sets.newConcurrentHashSet();
  public final Set<SkyKey> enqueued = Sets.newConcurrentHashSet();
  public final Set<SkyKey> evaluated = Sets.newConcurrentHashSet();

  public TrackingProgressReceiver(boolean checkEvaluationResults) {
    this.checkEvaluationResults = checkEvaluationResults;
  }

  @Override
  public void dirtied(SkyKey skyKey, DirtyType dirtyType) {
    runInvalidationCallbackIfPresent();
    dirty.add(skyKey);
    Preconditions.checkState(!deleted.contains(skyKey), skyKey);
  }

  @Override
  public void deleted(SkyKey skyKey) {
    runInvalidationCallbackIfPresent();
    dirty.remove(skyKey);
    deleted.add(skyKey);
  }

  private void runInvalidationCallbackIfPresent() {
    Runnable callback = nextInvalidationCallback.getAndSet(null);
    if (callback != null) {
      callback.run();
    }
  }

  @Override
  public void enqueueing(SkyKey skyKey) {
    enqueued.add(skyKey);
  }

  @Override
  public void evaluated(
      SkyKey skyKey,
      EvaluationState state,
      @Nullable SkyValue value,
      @Nullable ErrorInfo error,
      @Nullable GroupedDeps directDeps) {
    evaluated.add(skyKey);
    if (checkEvaluationResults && state.succeeded()) {
      deleted.remove(skyKey);
      if (!state.versionChanged()) {
        dirty.remove(skyKey);
      }
    }
  }

  public void clear() {
    dirty.clear();
    deleted.clear();
    enqueued.clear();
    evaluated.clear();
  }

  void setNextInvalidationCallback(Runnable runnable) {
    Runnable oldCallback = nextInvalidationCallback.getAndSet(runnable);
    Preconditions.checkState(
        oldCallback == null, "Overwriting a left-over callback: %s", oldCallback);
  }
}
