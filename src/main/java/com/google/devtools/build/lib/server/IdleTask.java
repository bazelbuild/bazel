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
package com.google.devtools.build.lib.server;

import java.time.Duration;

/** A task to be executed by {@link IdleTaskManager} while the server is idle. */
public interface IdleTask {
  /**
   * Returns how long to remain idle before executing the task.
   *
   * <p>The {@link IdleTaskManager} will wait for at least this long, but may delay execution
   * further if many cleanup tasks were registered.
   */
  default Duration delay() {
    return Duration.ZERO;
  }

  /**
   * Starts executing the task.
   *
   * <p>This method will be called in a dedicated thread. It should be designed to return early in
   * response to an interrupt, but is not required to do so. Either way, {@link IdleTaskManager}
   * must wait for it to return before leaving the idle period.
   */
  void run();
}
