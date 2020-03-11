// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android;

import com.android.annotations.VisibleForTesting;
import com.android.annotations.concurrency.Immutable;
import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.android.AndroidDataMerger.SourceChecker;
import java.io.IOException;
import java.util.Objects;
import java.util.ServiceLoader;

/**
 * Represents a conflict of two DataResources or DataAssets.
 *
 * <p>For resources, the dataKey is the FullyQualifiedName; Assets use the RelativeAssetPath.
 */
@Immutable
public class MergeConflict {

  private static final class LazyHolder {
    static final ImmutableList<MergeConflictExempter> MERGE_CONFLICT_EXEMPTERS =
        ImmutableList.copyOf(ServiceLoader.load(MergeConflictExempter.class));
  }

  private static final String CONFLICT_MESSAGE =
      "\n\u001B[31mCONFLICT:\u001B[0m"
          + " %s is provided with ambiguous priority from:\n\t%s\n\t%s";

  private final DataKey dataKey;
  private final DataValue primary;
  private final DataValue overwritten;

  private MergeConflict(DataKey dataKey, DataValue sortedFirst, DataValue sortedSecond) {
    this.dataKey = dataKey;
    this.primary = sortedFirst;
    this.overwritten = sortedSecond;
  }

  /**
   * Creates a MergeConflict between two DataResources.
   *
   * <p>The {@link DataKey} must match the first.dataKey() and second .dataKey().
   *
   * @param dataKey The dataKey name that both DataResources share.
   * @param primary The DataResource that will be used in the graph.
   * @param overwritten The DataResource that is replaced.
   * @return A new MergeConflict.
   */
  public static MergeConflict between(DataKey dataKey, DataValue primary, DataValue overwritten) {
    Preconditions.checkNotNull(dataKey);
    return of(dataKey, primary, overwritten);
  }

  @VisibleForTesting
  static MergeConflict of(DataKey key, DataValue primary, DataValue overwritten) {
    // Make sure the paths are always ordered.
    DataValue sortedFirst =
        primary.source().compareTo(overwritten.source()) > 0 ? primary : overwritten;
    DataValue sortedSecond = sortedFirst != primary ? primary : overwritten;
    return new MergeConflict(key, sortedFirst, sortedSecond);
  }

  public String toConflictMessage() {
    return String.format(
        CONFLICT_MESSAGE,
        dataKey.toPrettyString(),
        primary.asConflictString(),
        overwritten.asConflictString());
  }

  public DataKey dataKey() {
    return dataKey;
  }

  DataValue primary() {
    return primary;
  }

  DataValue overwritten() {
    return overwritten;
  }

  boolean isValidWith(SourceChecker checker) throws IOException {
    return dataKey.shouldDetectConflicts()
        && !primary.valueEquals(overwritten)
        && primary.compareMergePriorityTo(overwritten) == 0
        // TODO: SourceChecker can probably be removed, since the only no-op use is from AAR
        // generation (which shouldn't need to do these checks anyway).
        && !checker.checkEquality(primary.source(), overwritten.source())
        && !isExempted();
  }

  private boolean isExempted() {
    for (MergeConflictExempter mce : LazyHolder.MERGE_CONFLICT_EXEMPTERS) {
      if (mce.shouldAllow(dataKey, primary, overwritten)) {
        return true;
      }
    }
    return false;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("dataKey", dataKey)
        .add("primary", primary)
        .add("overwritten", overwritten)
        .toString();
  }

  @Override
  public boolean equals(Object other) {
    if (this == other) {
      return true;
    }
    if (!(other instanceof MergeConflict)) {
      return false;
    }
    MergeConflict that = (MergeConflict) other;
    return Objects.equals(dataKey, that.dataKey)
        && Objects.equals(primary, that.primary)
        && Objects.equals(overwritten, that.overwritten);
  }

  @Override
  public int hashCode() {
    return Objects.hash(dataKey, primary, overwritten);
  }
}
