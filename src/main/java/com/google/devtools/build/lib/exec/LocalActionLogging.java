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

package com.google.devtools.build.lib.exec;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.ActionMetadata;
import com.google.devtools.build.lib.util.BlazeClock;

import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

/**
 * Utility class for logging time of locally executed actions.
 */
class LocalActionLogging {

  private static final Logger LOG = Logger.getLogger(LocalActionLogging.class.getName());

  private final ActionMetadata action;
  private final long startTime;
  private boolean isRunning = true;

  LocalActionLogging(ActionMetadata action) {
    this.action = action;
    startTime = BlazeClock.nanoTime();
    LOG.info("Started to run " + action.prettyPrint());
  }

  void finish() {
    Preconditions.checkState(isRunning, action);
    isRunning = false;
    LOG.info("Finished running " + action.prettyPrint()
        + " in " + TimeUnit.NANOSECONDS.toMillis(BlazeClock.nanoTime() - startTime) + "ms");
  }
}
