// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.profiler.memory;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.packages.AspectClass;
import com.google.devtools.build.lib.packages.RuleClass;

/** Thread-local variables that keep track of the current rule being configured. */
public final class CurrentRuleTracker {
  private static final ThreadLocal<RuleClass> currentRule = new ThreadLocal<>();
  private static final ThreadLocal<AspectClass> currentAspect = new ThreadLocal<>();
  private static boolean enabled;

  private CurrentRuleTracker() {}

  public static void setEnabled(boolean enabled) {
    CurrentRuleTracker.enabled = enabled;
  }

  /**
   * Sets the current rule being instantiated. Used for memory tracking.
   *
   * <p>You must call {@link CurrentRuleTracker#endConfiguredTarget()} after calling this.
   */
  public static void beginConfiguredTarget(RuleClass ruleClass) {
    if (!enabled) {
      return;
    }
    currentRule.set(ruleClass);
  }

  public static void endConfiguredTarget() {
    if (!enabled) {
      return;
    }
    currentRule.set(null);
  }

  /**
   * Sets the current aspect being instantiated. Used for memory tracking.
   *
   * <p>You must call {@link CurrentRuleTracker#endConfiguredAspect()} after calling this.
   */
  public static void beginConfiguredAspect(AspectClass aspectClass) {
    if (!enabled) {
      return;
    }
    currentAspect.set(aspectClass);
  }

  public static void endConfiguredAspect() {
    if (!enabled) {
      return;
    }
    currentAspect.set(null);
  }

  public static RuleClass getRule() {
    Preconditions.checkState(enabled);
    return currentRule.get();
  }

  public static AspectClass getAspect() {
    Preconditions.checkState(enabled);
    return currentAspect.get();
  }
}
