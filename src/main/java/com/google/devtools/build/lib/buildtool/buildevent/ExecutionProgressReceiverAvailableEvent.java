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
package com.google.devtools.build.lib.buildtool.buildevent;

import com.google.devtools.build.lib.buildtool.ExecutionProgressReceiver;

/**
 * Event that is raised after the creation of a progress receiver in the execution phase. The event
 * makes the progress receiver available.
 */
public class ExecutionProgressReceiverAvailableEvent {
  private final ExecutionProgressReceiver executionProgressReceiver;

  public ExecutionProgressReceiverAvailableEvent(
      ExecutionProgressReceiver executionProgressReceiver) {
    this.executionProgressReceiver = executionProgressReceiver;
  }

  public ExecutionProgressReceiver getExecutionProgressReceiver() {
    return executionProgressReceiver;
  }
}
