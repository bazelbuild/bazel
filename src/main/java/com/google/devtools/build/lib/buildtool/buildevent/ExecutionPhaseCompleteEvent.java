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

package com.google.devtools.build.lib.buildtool.buildevent;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.view.TransitiveInfoCollection;

import java.util.Collection;

/**
 * This event is fired after the execution phase is complete.
 */
public class ExecutionPhaseCompleteEvent {

  private final Collection<TransitiveInfoCollection> targets;
  private final boolean keepGoing;
  private final long timeInMs;

  /**
   * Construct the event.
   * @param targets The set of active targets that remain.
   * @param keepGoing whether keepGoing is enabled
   * @param timeInMs time for execution phase in milliseconds.
   */
  public ExecutionPhaseCompleteEvent(Collection<? extends TransitiveInfoCollection> targets,
      boolean keepGoing, long timeInMs) {
    this.timeInMs = timeInMs;
    this.targets = ImmutableList.copyOf(targets);
    this.keepGoing = keepGoing;
  }

  /**
   * @return The set of active targets remaining, which is a subset
   *     of the targets we attempted to build.
   */
  public Collection<TransitiveInfoCollection> getTargets() {
    return targets;
  }

  /**
   * @return The value of the keepGoing flag.
   */
  public boolean keepGoing() {
    return keepGoing;
  }

  public long getTimeInMs() {
    return timeInMs;
  }
}
