// Copyright 2015 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import javax.annotation.Nullable;

/**
 * Given a {@link SkyKey} and the previous {@link SkyValue} it had, returns whether this value is
 * up to date.
 */
public interface SkyValueDirtinessChecker {
  /**
   * Returns the result of checking whether this key's value is up to date, or null if this
   * dirtiness checker does not apply to this key. If non-null, this answer is assumed to be
   * definitive.
   */
  @Nullable DirtyResult maybeCheck(SkyKey key, SkyValue oldValue, TimestampGranularityMonitor tsgm);

  /** An encapsulation of the result of checking to see if a value is up to date. */
  class DirtyResult {
    /** The external value is unchanged from the value in the graph. */
    public static final DirtyResult NOT_DIRTY = new DirtyResult(false, null);
    /**
     * The external value is different from the value in the graph, but the new value is not known.
     */
    public static final DirtyResult DIRTY = new DirtyResult(true, null);

    /**
     * Creates a DirtyResult indicating that the external value is {@param newValue}, which is
     * different from the value in the graph,
     */
    public static DirtyResult dirtyWithNewValue(SkyValue newValue) {
      return new DirtyResult(true, newValue);
    }

    private final boolean isDirty;
    @Nullable private final SkyValue newValue;

    private DirtyResult(boolean isDirty, @Nullable SkyValue newValue) {
      this.isDirty = isDirty;
      this.newValue = newValue;
    }

    boolean isDirty() {
      return isDirty;
    }

    /**
     * If {@code isDirty()}, then either returns the new value for the value or {@code null} if
     * the new value wasn't computed. In the case where the value is dirty and a new value is
     * available, then the new value can be injected into the skyframe graph. Otherwise, the value
     * should simply be invalidated.
     */
    @Nullable
    SkyValue getNewValue() {
      Preconditions.checkState(isDirty(), newValue);
      return newValue;
    }
  }
}
