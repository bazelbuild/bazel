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

import java.nio.file.Path;
import java.util.Objects;

/**
 * Represents a conflict of two DataResources or DataAssets.
 *
 * For resources, the dataKey is the FullyQualifiedName; Assets use the RelativeAssetPath.
 */
@Immutable
public class MergeConflict {
  static final String CONFLICT_MESSAGE = "%s is provided from %s and %s";
  private final DataKey dataKey;
  private final Path first;
  private final Path second;

  private MergeConflict(DataKey dataKey, Path first, Path second) {
    this.dataKey = dataKey;
    this.first = first;
    this.second = second;
  }

  /**
   * Creates a MergeConflict between two DataResources.
   *
   * The {@link DataKey} must match the first.dataKey() and second
   * .dataKey().
   *
   * @param dataKey The dataKey name that both DataResources share.
   * @param first The first DataResource.
   * @param second The second DataResource.
   * @return A new MergeConflict.
   */
  public static MergeConflict between(DataKey dataKey, DataResource first, DataResource second) {
    Preconditions.checkNotNull(dataKey);
    Preconditions.checkArgument(dataKey.equals(first.dataKey()));
    Preconditions.checkArgument(dataKey.equals(second.dataKey()));
    return of(dataKey, first.source(), second.source());
  }

  /**
   * Creates a MergeConflict between two DataResources.
   *
   * The {@link DataKey} must match the first.dataKey() and second
   * .dataKey().
   *
   * @param dataKey The dataKey name that both DataResources share.
   * @param first The first DataResource.
   * @param second The second DataResource.
   * @return A new MergeConflict.
   */
  public static MergeConflict between(DataKey dataKey, DataAsset first, DataAsset second) {
    Preconditions.checkNotNull(dataKey);
    Preconditions.checkArgument(dataKey.equals(first.dataKey()));
    Preconditions.checkArgument(dataKey.equals(second.dataKey()));
    return of(dataKey, first.source(), second.source());
  }

  @VisibleForTesting
  static MergeConflict of(DataKey key, Path first, Path second) {
    return new MergeConflict(key, first, second);
  }

  public String toConflictMessage() {
    return String.format(CONFLICT_MESSAGE, dataKey, first, second);
  }

  public DataKey dataKey() {
    return dataKey;
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
    return Objects.equals(first, that.first) && Objects.equals(second, that.second);
  }

  @Override
  public int hashCode() {
    return Objects.hash(first, second);
  }
}
