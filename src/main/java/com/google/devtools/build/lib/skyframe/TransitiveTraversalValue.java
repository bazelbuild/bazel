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
package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.util.Objects;

import javax.annotation.Nullable;

/**
 * A <i>transitive</i> target reference that, when built in skyframe, loads the entire
 * transitive closure of a target. Contains no information about the targets traversed.
 */
@Immutable
@ThreadSafe
public class TransitiveTraversalValue implements SkyValue {

  @Nullable
  private final NoSuchTargetException errorLoadingTarget;

  // Note that this value does not guarantee singleton-like reference equality for successful
  // {@link TransitiveTraversalValue}s because we use Java deserialization. Java deserialization can
  // create other instances.
  public static final TransitiveTraversalValue SUCCESSFUL_TRANSITIVE_TRAVERSAL_VALUE =
      new TransitiveTraversalValue(null);

  public static TransitiveTraversalValue unsuccessfulTransitiveTraversal(
      NoSuchTargetException errorLoadingTarget) {
    return new TransitiveTraversalValue(Preconditions.checkNotNull(errorLoadingTarget));
  }

  private TransitiveTraversalValue(@Nullable NoSuchTargetException errorLoadingTarget) {
    this.errorLoadingTarget = errorLoadingTarget;
  }

  /** Returns the error, if any, from loading the target. */
  @Nullable
  public NoSuchTargetException getErrorLoadingTarget() {
    return errorLoadingTarget;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (!(o instanceof TransitiveTraversalValue)) {
      return false;
    }
    TransitiveTraversalValue that = (TransitiveTraversalValue) o;
    return Objects.equals(this.errorLoadingTarget, that.errorLoadingTarget);
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(errorLoadingTarget);
  }

  @ThreadSafe
  public static SkyKey key(Label label) {
    return new SkyKey(SkyFunctions.TRANSITIVE_TRAVERSAL, label);
  }
}
