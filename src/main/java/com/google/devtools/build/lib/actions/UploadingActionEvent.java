// Copyright 2026 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;

/**
 * Notifies that an in-flight action is uploading inputs to the remote cache.
 *
 * @param action Gets the metadata associated with the action.
 * @param strategy Gets the name of the strategy on which the action is uploading.
 */
public record UploadingActionEvent(ActionExecutionMetadata action, String strategy)
    implements Postable {
  public UploadingActionEvent {
    checkNotNull(action, "action");
    checkNotNull(strategy, "strategy");
  }

  public static UploadingActionEvent create(ActionExecutionMetadata action, String strategy) {
    return new UploadingActionEvent(action, strategy);
  }
}
