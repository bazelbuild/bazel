// Copyright 2023 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.skyframe.NodeEntry.DirtyType;
import com.google.errorprone.annotations.ForOverride;

/**
 * Test utility that funnels both {@link #dirtied} and {@link #deleted} to a single {@link
 * #invalidated} method.
 */
public abstract class InvalidationProgressReceiver implements EvaluationProgressReceiver {

  @Override
  public final void dirtied(SkyKey skyKey, DirtyType dirtyType) {
    invalidated(skyKey, InvalidationState.DIRTY);
  }

  @Override
  public final void deleted(SkyKey skyKey) {
    invalidated(skyKey, InvalidationState.DELETED);
  }

  /** New state of the value entry after invalidation. */
  public enum InvalidationState {
    /** The value is dirty, although it might get re-validated again. */
    DIRTY,
    /** The value is dirty and got deleted, cannot get re-validated again. */
    DELETED,
  }

  @ForOverride
  protected abstract void invalidated(SkyKey skyKey, InvalidationState state);
}
