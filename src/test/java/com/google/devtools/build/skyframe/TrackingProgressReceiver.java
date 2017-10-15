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

import com.google.common.base.Supplier;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.util.Preconditions;
import java.util.Set;

/**
 * A testing utility to keep track of evaluation.
 */
public class TrackingProgressReceiver
    extends EvaluationProgressReceiver.NullEvaluationProgressReceiver {
  public final Set<SkyKey> dirty = Sets.newConcurrentHashSet();
  public final Set<SkyKey> deleted = Sets.newConcurrentHashSet();
  public final Set<SkyKey> enqueued = Sets.newConcurrentHashSet();
  public final Set<SkyKey> evaluated = Sets.newConcurrentHashSet();

  @Override
  public void invalidated(SkyKey skyKey, InvalidationState state) {
    switch (state) {
      case DELETED:
        dirty.remove(skyKey);
        deleted.add(skyKey);
        break;
      case DIRTY:
        dirty.add(skyKey);
        Preconditions.checkState(!deleted.contains(skyKey));
        break;
      default:
        throw new IllegalStateException();
    }
  }

  @Override
  public void enqueueing(SkyKey skyKey) {
    enqueued.add(skyKey);
  }

  @Override
  public void evaluated(SkyKey skyKey, Supplier<SkyValue> skyValueSupplier, EvaluationState state) {
    evaluated.add(skyKey);
    if (skyValueSupplier.get() != null) {
      deleted.remove(skyKey);
      if (state.equals(EvaluationState.CLEAN)) {
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
}
