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
package com.google.devtools.build.lib.runtime;

import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.analysis.NoBuildEvent;
import com.google.devtools.build.lib.analysis.NoBuildRequestFinishedEvent;

/**
 * A passive version of {@link ExperimentalEventHandler}.
 *
 * Given an {@link ExperimentalEventHandler}, generate a wrapper class, that
 * only subscribes to events that only modify its internal state, but do not
 * produce any output.
 */
public class PassiveExperimentalEventHandler {
  private final ExperimentalEventHandler eventHandler;

  public PassiveExperimentalEventHandler(ExperimentalEventHandler eventHandler) {
    this.eventHandler = eventHandler;
  }

  @Subscribe
  public void noBuild(NoBuildEvent event) {
    eventHandler.noBuild(event);
  }

  @Subscribe
  public void noBuildFinished(NoBuildRequestFinishedEvent event) {
    eventHandler.noBuildFinished(event);
  }
}
