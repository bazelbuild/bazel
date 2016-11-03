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

import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;

import com.android.annotations.VisibleForTesting;
import com.android.annotations.concurrency.Immutable;

import java.util.Objects;

/**
 * Represents a conflict of two DataResources or DataAssets.
 *
 * For resources, the dataKey is the FullyQualifiedName; Assets use the RelativeAssetPath.
 */
@Immutable
public class MergeConflict {
  private static final String CONFLICT_MESSAGE = "\n\u001B[31mCONFLICT:\u001B[0m"
          + " %s is provided with ambiguous priority from:\n\t%s\n\t%s";

  private final DataKey dataKey;
  private final DataValue first;
  private final DataValue second;

  private MergeConflict(DataKey dataKey, DataValue sortedFirst, DataValue sortedSecond) {
    this.dataKey = dataKey;
    this.first = sortedFirst;
    this.second = sortedSecond;
  }

  /**
   * Creates a MergeConflict between two DataResources.
   *
   * The {@link DataKey} must match the first.dataKey() and second .dataKey().
   *
   * @param dataKey The dataKey name that both DataResources share.
   * @param first The first DataResource.
   * @param second The second DataResource.
   * @return A new MergeConflict.
   */
  public static MergeConflict between(DataKey dataKey, DataValue first, DataValue second) {
    Preconditions.checkNotNull(dataKey);
    return of(dataKey, first, second);
  }

  @VisibleForTesting
  static MergeConflict of(DataKey key, DataValue first, DataValue second) {
    // Make sure the paths are always ordered.
    DataValue sortedFirst = first.source().compareTo(second.source()) > 0 ? first : second;
    DataValue sortedSecond = sortedFirst != first ? first : second;
    return new MergeConflict(key, sortedFirst, sortedSecond);
  }

  public String toConflictMessage() {
    return String.format(
        CONFLICT_MESSAGE, dataKey.toPrettyString(), first.source().getPath(),
        second.source().getPath());
  }

  public DataKey dataKey() {
    return dataKey;
  }

  DataValue first() {
    return first;
  }

  DataValue second() {
    return second;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("dataKey", dataKey)
        .add("first", first)
        .add("second", second)
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
        && Objects.equals(first, that.first)
        && Objects.equals(second, that.second);
  }

  @Override
  public int hashCode() {
    return Objects.hash(dataKey, first, second);
  }
}
