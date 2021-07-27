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
package com.google.devtools.build.lib.actions;

import com.google.auto.value.AutoValue;
import com.google.devtools.build.lib.events.ExtendedEventHandler.ProgressLike;

/** Notifications for the uploads to the remote server. */
@AutoValue
public abstract class RemoteUploadEvent implements ProgressLike {

  public static RemoteUploadEvent create(String resourceId, String progress, boolean finished) {
    return new AutoValue_RemoteUploadEvent(resourceId, progress, finished);
  }

  /** The id that uniquely determines the resource being uploaded among all events within an build. */
  public abstract String resourceId();

  /** Human readable description of the upload progress. */
  public abstract String progress();

  /** Whether the upload progress reported about is finished already. */
  public abstract boolean finished();
}
