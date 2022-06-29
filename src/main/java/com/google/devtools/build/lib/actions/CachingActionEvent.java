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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.auto.value.AutoValue;
import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;

/** Notifies that an in-flight action is checking the cache. */
@AutoValue
public abstract class CachingActionEvent implements Postable {

  public static CachingActionEvent create(ActionExecutionMetadata action, String strategy) {
    return new AutoValue_CachingActionEvent(
        action, checkNotNull(strategy, "Strategy names are not optional"));
  }

  /** Gets the metadata associated with the action. */
  public abstract ActionExecutionMetadata action();

  /** Gets the name of the strategy on which the action is caching. */
  public abstract String strategy();
}
