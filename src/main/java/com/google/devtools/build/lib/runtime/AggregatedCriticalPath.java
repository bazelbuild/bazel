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

package com.google.devtools.build.lib.runtime;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;

/**
 * Aggregates all the critical path components in one object. This allows us to easily access the
 * components data and have a proper toString().
 */
public class AggregatedCriticalPath<T extends AbstractCriticalPathComponent<?>> {

  private final long totalTime;
  private final ImmutableList<T> criticalPathComponents;

  protected AggregatedCriticalPath(long totalTime, ImmutableList<T> criticalPathComponents) {
    this.totalTime = totalTime;
    this.criticalPathComponents = criticalPathComponents;
  }

  /** Total wall time in ms spent running the critical path actions. */
  public long totalTime() {
    return totalTime;
  }

  /** Returns a list of all the component stats for the critical path. */
  public ImmutableList<T> components() {
    return criticalPathComponents;
  }

  @Override
  public String toString() {
    return toString(false);
  }

  /**
   * Returns a summary version of the critical path stats that omits stats that are not useful
   * to the user.
   */
  public String toStringSummary() {
    return toString(true);
  }

  private String toString(boolean summary) {
    StringBuilder sb = new StringBuilder("Critical Path: ");
    double totalMillis = totalTime;
    sb.append(String.format("%.2f", totalMillis / 1000.0));
    sb.append("s");
    if (summary || criticalPathComponents.isEmpty()) {
      return sb.toString();
    }
    sb.append("\n  ");
    Joiner.on("\n  ").appendTo(sb, criticalPathComponents);
    return sb.toString();
  }
}

