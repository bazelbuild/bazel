// Copyright 2017 The Bazel Authors. All rights reserved.
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
 * A marker interface for classes that provide services for actions during execution.
 *
 * <p>Interfaces extending this one should also be annotated with {@link ActionContextMarker}.
 */
public interface ActionContext {
  /**
   * Called when the executor is constructed. The parameter contains all the contexts that were
   * selected for this execution phase.
   */
  default void executorCreated(Iterable<ActionContext> usedContexts) throws ExecutorInitException {}
}
