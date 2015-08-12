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

import com.google.common.base.Optional;
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
   * Returns
   * <ul>
   *   <li>{@code null}, if the checker can't handle {@code key}.
   *   <li>{@code Optional.<SkyValue>absent()} if the checker can handle {@code key} but was unable
   *       to create a new value.
   *   <li>{@code Optional.<SkyValue>of(v)} if the checker can handle {@code key} and the new value
   *       should be {@code v}.
   * </ul>
   */
  @Nullable
  public abstract Optional<SkyValue> maybeCreateNewValue(SkyKey key,
      TimestampGranularityMonitor tsgm);

  /**
   * Returns the result of checking whether this key's value is up to date, or null if this
   * dirtiness checker does not apply to this key. If non-null, this answer is assumed to be
   * definitive.
   */
  @Nullable
  public DirtyResult maybeCheck(SkyKey key, @Nullable SkyValue oldValue,
      TimestampGranularityMonitor tsgm) {
    Optional<SkyValue> newValueMaybe = maybeCreateNewValue(key, tsgm);
    if (newValueMaybe == null) {
      return null;
    }
    if (!newValueMaybe.isPresent()) {
      return DirtyResult.dirty(oldValue);
    }
    SkyValue newValue = Preconditions.checkNotNull(newValueMaybe.get(), key);
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
      return new DirtyResult(/*dirty=*/false, oldValue,  /*newValue=*/null);
    }

    /**
     * Creates a DirtyResult indicating that external value is different from the value in the
     * graph, but this new value is not known.
     */
    public static DirtyResult dirty(@Nullable SkyValue oldValue) {
      return new DirtyResult(/*dirty=*/true, oldValue, /*newValue=*/null);
    }

    /**
     * Creates a DirtyResult indicating that the external value is {@code newValue}, which is
     * different from the value in the graph,
     */
    public static DirtyResult dirtyWithNewValue(@Nullable SkyValue oldValue, SkyValue newValue) {
      return new DirtyResult(/*dirty=*/true, oldValue, newValue);
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

    boolean isDirty() {
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
