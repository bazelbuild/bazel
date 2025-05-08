// Copyright 2024 The Bazel Authors. All rights reserved.
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

import java.util.Comparator;

/** Used with dump --rules and BEP SkyFrameMetrics. */
public class SkyKeyStats {

  public static final Comparator<SkyKeyStats> BY_COUNT_DESC =
      Comparator.comparing(SkyKeyStats::getCount).reversed();

  private final String key;
  private final String name;
  private long count;
  private long actionCount;

  public SkyKeyStats(String key, String name) {
    this.key = key;
    this.name = name;
  }

  public void countWithActions(long actionCount) {
    this.count++;
    this.actionCount += actionCount;
  }

  public void count() {
    this.count++;
  }

  /** Returns a key that uniquely identifies this rule or aspect. */
  public String getKey() {
    return key;
  }

  /** Returns a name for the rule or aspect. */
  public String getName() {
    return name;
  }

  /** Returns the instance count of this rule or aspect class. */
  public long getCount() {
    return count;
  }

  /** Returns the total action count of all instance of this rule or aspect class. */
  public long getActionCount() {
    return actionCount;
  }
}
