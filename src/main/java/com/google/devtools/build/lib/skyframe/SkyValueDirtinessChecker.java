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

import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.Version;
import java.util.Objects;
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
  public abstract SkyValue createNewValue(
      SkyKey key, SyscallCache syscallCache, @Nullable TimestampGranularityMonitor tsgm);

  /**
   * Returns the max transitive source version (mtsv) of a {@link SkyKey} for its new {@link
   * SkyValue}.
   */
  @Nullable
  public Version getMaxTransitiveSourceVersionForNewValue(SkyKey key, SkyValue value) {
    return null;
  }

  /**
   * Returns whether it is ok for this {@link SkyValueDirtinessChecker} to return a null max
   * transitive source version. If this method returns false, a null mtsv would indicate an {@link
   * java.io.IOException} was thrown.
   */
  public boolean nullMaxTransitiveSourceVersionOk() {
    return true;
  }

  /**
   * If {@code applies(key)}, returns the result of checking whether this key's value is up to date.
   */
  public DirtyResult check(
      SkyKey key,
      @Nullable SkyValue oldValue,
      @Nullable Version oldMtsv,
      SyscallCache syscallCache,
      @Nullable TimestampGranularityMonitor tsgm) {
    SkyValue newValue = createNewValue(key, syscallCache, tsgm);
    if (newValue == null) {
      return DirtyResult.dirty();
    }
    return newValue.equals(oldValue)
        ? DirtyResult.notDirty()
        : DirtyResult.dirtyWithNewValue(newValue);
  }

  /** An encapsulation of the result of checking to see if a value is up to date. */
  // TODO(b/228090733) - support old source versions for dirtiness checking
  public static final class DirtyResult {
    private static final DirtyResult NOT_DIRTY =
        new DirtyResult(
            /* isDirty= */ false, /* newValue= */ null, /* newMaxTransitiveSourceVersion= */ null);
    private static final DirtyResult DIRTY =
        new DirtyResult(
            /* isDirty= */ true, /* newValue= */ null, /* newMaxTransitiveSourceVersion= */ null);

    /**
     * Creates a DirtyResult indicating that the external value is the same as the value in the
     * graph.
     */
    public static DirtyResult notDirty() {
      return NOT_DIRTY;
    }

    /**
     * Creates a DirtyResult indicating that external value is different from the value in the
     * graph, but this new value is not known.
     */
    public static DirtyResult dirty() {
      return DIRTY;
    }

    /**
     * Creates a DirtyResult indicating that the external value is {@code newValue}, which is
     * different from the value in the graph,
     */
    public static DirtyResult dirtyWithNewValue(SkyValue newValue) {
      return new DirtyResult(
          /* isDirty= */ true, newValue, /* newMaxTransitiveSourceVersion= */ null);
    }

    public static DirtyResult dirtyWithNewValueAndMaxTransitiveSourceVersion(
        SkyValue newValue, Version newMaxTransitiveSourceVersion) {
      return new DirtyResult(/* isDirty= */ true, newValue, newMaxTransitiveSourceVersion);
    }

    private final boolean isDirty;
    @Nullable private final SkyValue newValue;
    @Nullable private final Version newMaxTransitiveSourceVersion;

    private DirtyResult(
        boolean isDirty,
        @Nullable SkyValue newValue,
        @Nullable Version newMaxTransitiveSourceVersion) {
      this.isDirty = isDirty;
      this.newValue = newValue;
      this.newMaxTransitiveSourceVersion = newMaxTransitiveSourceVersion;
    }

    public boolean isDirty() {
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

    /**
     * Returns the max transitive source version for the new value or {@code null}.
     *
     * <p>Can only be called if the result {@code isDirty()}.
     */
    @Nullable
    Version getNewMaxTransitiveSourceVersion() {
      Preconditions.checkState(isDirty(), newValue);
      return newMaxTransitiveSourceVersion;
    }

    @Override
    public int hashCode() {
      return Objects.hashCode(newValue)
          + (isDirty ? 13 : 0)
          + Objects.hashCode(newMaxTransitiveSourceVersion);
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (!(obj instanceof DirtyResult)) {
        return false;
      }
      DirtyResult that = (DirtyResult) obj;
      return this.isDirty == that.isDirty
          && Objects.equals(this.newValue, that.newValue)
          && Objects.equals(this.newMaxTransitiveSourceVersion, that.newMaxTransitiveSourceVersion);
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this)
          .add("isDirty", isDirty)
          .add("newValue", newValue)
          .add("newMaxTransitiveSourceVersion", newMaxTransitiveSourceVersion)
          .toString();
    }
  }
}
