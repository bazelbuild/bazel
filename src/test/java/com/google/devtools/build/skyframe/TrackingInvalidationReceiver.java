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

import com.google.common.base.Preconditions;
import com.google.common.collect.Sets;

import java.util.Set;

/**
 * A testing utility to keep track of evaluation.
 */
public class TrackingInvalidationReceiver implements EvaluationProgressReceiver {
  public final Set<SkyValue> dirty = Sets.newConcurrentHashSet();
  public final Set<SkyValue> deleted = Sets.newConcurrentHashSet();
  public final Set<SkyKey> enqueued = Sets.newConcurrentHashSet();
  public final Set<SkyKey> evaluated = Sets.newConcurrentHashSet();

  @Override
  public void invalidated(SkyValue value, InvalidationState state) {
    switch (state) {
      case DELETED:
        dirty.remove(value);
        deleted.add(value);
        break;
      case DIRTY:
        dirty.add(value);
        Preconditions.checkState(!deleted.contains(value));
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
  public void evaluated(SkyKey skyKey, SkyValue value, EvaluationState state) {
    evaluated.add(skyKey);
    if (value != null) {
      dirty.remove(value);
      deleted.remove(value);
    }
  }

  public void clear() {
    dirty.clear();
    deleted.clear();
    enqueued.clear();
    evaluated.clear();
  }
}
