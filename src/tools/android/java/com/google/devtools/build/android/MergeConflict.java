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

import java.util.Objects;

/**
 * Represents a conflict of two DataResources that share the same FullyQualifiedName.
 */
public class MergeConflict {
  static final String CONFLICT_MESSAGE = "%s is provided from %s and %s";
  private final FullyQualifiedName fullyQualifiedName;
  private final DataResource first;
  private final DataResource second;

  private MergeConflict(
      FullyQualifiedName fullyQualifiedName, DataResource first, DataResource second) {
    this.fullyQualifiedName = fullyQualifiedName;
    this.first = first;
    this.second = second;
  }

  public static MergeConflict between(
      FullyQualifiedName fullyQualifiedName, DataResource first, DataResource second) {
    return new MergeConflict(fullyQualifiedName, first, second);
  }

  public String toConflictMessage() {
    return String.format(CONFLICT_MESSAGE, fullyQualifiedName, first.source(), second.source());
  }

  public FullyQualifiedName fullyQualifiedName() {
    return fullyQualifiedName;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this).add("first", first).add("second", second).toString();
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
