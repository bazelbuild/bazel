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
package com.google.devtools.build.lib.analysis;

import com.google.common.base.Preconditions;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.skyframe.AspectValue;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.skyframe.SkyValue;

/**
 * This event is fired as soon as a top-level aspect is either built or fails.
 */
public class AspectCompleteEvent implements SkyValue {
  private final AspectValue aspectValue;
  private final NestedSet<Label> rootCauses;

  private AspectCompleteEvent(AspectValue aspectValue, NestedSet<Label> rootCauses) {
    this.aspectValue = aspectValue;
    this.rootCauses =
        (rootCauses == null) ? NestedSetBuilder.<Label>emptySet(Order.STABLE_ORDER) : rootCauses;
  }

  /**
   * Construct a successful target completion event.
   */
  public static AspectCompleteEvent createSuccessful(AspectValue value) {
    return new AspectCompleteEvent(value, null);
  }

  /**
   * Construct a target completion event for a failed target, with the given non-empty root causes.
   */
  public static AspectCompleteEvent createFailed(AspectValue value, NestedSet<Label> rootCauses) {
    Preconditions.checkArgument(!Iterables.isEmpty(rootCauses));
    return new AspectCompleteEvent(value, rootCauses);
  }

  /**
   * Returns the target associated with the event.
   */
  public AspectValue getAspectValue() {
    return aspectValue;
  }

  /**
   * Determines whether the target has failed or succeeded.
   */
  public boolean failed() {
    return !rootCauses.isEmpty();
  }

  /**
   * Get the root causes of the target. May be empty.
   */
  public Iterable<Label> getRootCauses() {
    return rootCauses;
  }
}
