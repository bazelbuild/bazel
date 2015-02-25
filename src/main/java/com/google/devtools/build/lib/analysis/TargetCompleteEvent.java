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

package com.google.devtools.build.lib.analysis;

import com.google.common.base.Preconditions;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.skyframe.SkyValue;

/**
 * This event is fired as soon as a target is either built or fails.
 */
public final class TargetCompleteEvent implements SkyValue {

  private final ConfiguredTarget target;
  private final NestedSet<Label> rootCauses;

  private TargetCompleteEvent(ConfiguredTarget target, NestedSet<Label> rootCauses) {
    this.target = target;
    this.rootCauses = (rootCauses == null)
        ? NestedSetBuilder.<Label>emptySet(Order.STABLE_ORDER)
        : rootCauses;
  }

  /**
   * Construct a successful target completion event.
   */
  public static TargetCompleteEvent createSuccessful(ConfiguredTarget ct) {
    return new TargetCompleteEvent(ct, null);
  }

  /**
   * Construct a target completion event for a failed target, with the given non-empty root causes.
   */
  public static TargetCompleteEvent createFailed(ConfiguredTarget ct, NestedSet<Label> rootCauses) {
    Preconditions.checkArgument(!Iterables.isEmpty(rootCauses));
    return new TargetCompleteEvent(ct, rootCauses);
  }

  /**
   * Returns the target associated with the event.
   */
  public ConfiguredTarget getTarget() {
    return target;
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
