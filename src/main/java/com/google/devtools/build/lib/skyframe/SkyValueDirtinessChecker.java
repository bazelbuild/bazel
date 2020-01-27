// Copyright 2015 The Bazel Authors. All rights reserved.
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
public abstract class SkyValueDirtinessChecker {
  /**
   * Returns {@code true} iff the checker can handle {@code key}. Can only be true if {@code
   * key.functionName().getHermeticity() == FunctionHermeticity.NONHERMETIC}.
   */
  public abstract boolean applies(SkyKey key);

  /**
   * If {@code applies(key)}, returns the new value for {@code key} or {@code null} if the checker
   * was unable to create a new value.
   */
  @Nullable
  public abstract SkyValue createNewValue(SkyKey key, @Nullable TimestampGranularityMonitor tsgm);

  /**
   * If {@code applies(key)}, returns the result of checking whether this key's value is up to date.
   */
  public DirtyResult check(SkyKey key, @Nullable SkyValue oldValue,
      @Nullable TimestampGranularityMonitor tsgm) {
    SkyValue newValue = createNewValue(key, tsgm);
    if (newValue == null) {
      return DirtyResult.dirty(oldValue);
    }
    return newValue.equals(oldValue)
        ? DirtyResult.notDirty(oldValue)
        : DirtyResult.dirtyWithNewValue(oldValue, newValue);
  }

  /** An encapsulation of the result of checking to see if a value is up to date. */
  public static class DirtyResult {
    /**
     * Creates a DirtyResult indicating that the external value is the same as the value in the
     * graph.
     */
    public static DirtyResult notDirty(SkyValue oldValue) {
      return new DirtyResult(/*isDirty=*/false, oldValue,  /*newValue=*/null);
    }

    /**
     * Creates a DirtyResult indicating that external value is different from the value in the
     * graph, but this new value is not known.
     */
    public static DirtyResult dirty(@Nullable SkyValue oldValue) {
      return new DirtyResult(/*isDirty=*/true, oldValue, /*newValue=*/null);
    }

    /**
     * Creates a DirtyResult indicating that the external value is {@code newValue}, which is
     * different from the value in the graph,
     */
    public static DirtyResult dirtyWithNewValue(@Nullable SkyValue oldValue, SkyValue newValue) {
      return new DirtyResult(/*isDirty=*/true, oldValue, newValue);
    }

    private final boolean isDirty;
    @Nullable private final SkyValue oldValue;
    @Nullable private final SkyValue newValue;

    private DirtyResult(boolean isDirty, @Nullable SkyValue oldValue,
        @Nullable SkyValue newValue) {
      this.isDirty = isDirty;
      this.oldValue = oldValue;
      this.newValue = newValue;
    }

    public boolean isDirty() {
      return isDirty;
    }

    @Nullable
    SkyValue getOldValue() {
      return oldValue;
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
