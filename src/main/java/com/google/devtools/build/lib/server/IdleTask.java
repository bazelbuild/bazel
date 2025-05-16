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

  /** The status of an idle task. */
  public enum Status {
    /** The task was never started. */
    NOT_STARTED,
    /** The task finished successfully. */
    SUCCESS,
    /** The task finished with an error. */
    FAILURE,
    /* The task was interrupted while running. */
    INTERRUPTED
  }

  /** The result of running an idle task. */
  public record Result(String name, Status status, Duration runningTime) {}

  /** A display name for the task. */
  String displayName();

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
   * Executes the task.
   *
   * <p>Implementations are encouraged to handle interruption promptly by throwing {@link
   * InterruptedException}, but are not required to do so, and may instead opt to complete normally.
   * Either way, the {@link IdleTaskManager} will wait for the task to finish before leaving the
   * idle period, so the task may assume that a subsequent command will not interfere with it.
   *
   * @throws InterruptedException if the task is interrupted
   * @throws IdleTaskException if the task fails for a reason other than interruption
   */
  void run() throws IdleTaskException, InterruptedException;
}
