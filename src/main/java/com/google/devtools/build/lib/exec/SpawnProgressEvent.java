// Copyright 2021 The Bazel Authors. All rights reserved.
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

import com.google.auto.value.AutoValue;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ActionProgressEvent;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.exec.SpawnRunner.ProgressStatus;

/** The {@link SpawnRunner} is making some progress. */
@AutoValue
public abstract class SpawnProgressEvent implements ProgressStatus {

  public static SpawnProgressEvent create(String resourceId, String progress, boolean finished) {
    return new AutoValue_SpawnProgressEvent(resourceId, progress, finished);
  }

  /** The id that uniquely determines the progress among all progress events for this spawn. */
  public abstract String progressId();

  /** Human readable description of the progress. */
  public abstract String progress();

  /** Whether the progress reported about is finished already. */
  public abstract boolean finished();

  @Override
  public void postTo(ExtendedEventHandler eventHandler, ActionExecutionMetadata action) {
    eventHandler.post(ActionProgressEvent.create(action, progressId(), progress(), finished()));
  }
}
