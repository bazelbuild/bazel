// Copyright 2014 The Bazel Authors. All rights reserved.
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

/**
 * An action which must know when it is skipped due to an action cache hit.
 *
 * Use should be rare, as the action graph is a functional model.
 */
public interface NotifyOnActionCacheHit extends Action {

  /**
   * Called when action has "cache hit", and therefore need not be executed.
   *
   * @param executor the executor
   */
  void actionCacheHit(Executor executor);
}
